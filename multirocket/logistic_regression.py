import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal, Union

import h5py
import numpy as np
import torch
import torch.nn.functional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


logger = logging.getLogger(__name__)


class LogisticRegression:

    def __init__(
        self,
        num_features,
        max_epochs=500,
        minibatch_size=256,
        validation_size=2**11,
        learning_rate=1e-4,
        patience_lr=5,  # 50 minibatches
        patience=10,  # 100 minibatches
        device=Literal["cuda", "cpu"],
        num_workers=0,
    ):
        self.name = "LogisticRegression"
        self.args = {
            "num_features": num_features,
            "validation_size": validation_size,
            "minibatch_size": minibatch_size,
            "lr": learning_rate,
            "max_epochs": max_epochs,
            "patience_lr": patience_lr,
            "patience": patience,
            "device": device,
            "num_workers": num_workers
        }
        self.num_workers = num_workers
        self.model = None
        self.device = torch.device(device)
        self.classes = None
        self.scaler = None
        self.num_classes = None
        self.scaler = StandardScaler(copy=False)

    def fit(self, x_train, y_train, stratify_var=None):
        self.classes = np.unique(y_train)
        self.num_classes = len(self.classes)

        if stratify_var is None:
            stratify_var = y_train

        num_outputs = self.num_classes if self.num_classes > 2 else 1

        logger.info(f'[{self.name}] Setting up logistic regression model')
        model = torch.nn.Sequential(torch.nn.Linear(self.args["num_features"], num_outputs)).to(self.device)

        if num_outputs == 1:
            loss_function = torch.nn.BCEWithLogitsLoss()
        else:
            loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, min_lr=1e-8, patience=self.args["patience_lr"]
        )

        # Setup dataset
        @dataclass
        class H5Dataset(torch.utils.data.Dataset):
            data_path: Path
            y: np.ndarray

            def __post_init__(self):
                self.x = h5py.File(self.data_path, 'r')['x']

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        if isinstance(x_train, Path):
            # Read tmp data file
            logger.info(f'Reading training data from tmp H5 file at {x_train}')
            f = h5py.File(x_train, 'r')
            x = f['x']
        else:
            x = x_train
        training_size = x.shape[0]
        if self.args["validation_size"] < training_size:
            logger.info("Splitting training data into training and validation sets.")
            idx_train, idx_val = train_test_split(range(training_size), test_size=self.args["validation_size"], stratify=stratify_var)

            # Compute scaling parameters by partial_fit on partial_samples samples at a time
            logger.info('Fitting scaler on training data')
            batch_size = 1000
            n_passes = len(idx_train) // batch_size
            x_batch = np.zeros((batch_size, x.shape[1]))
            s = time()
            for i in range(n_passes + 1):
                current_batch = batch_size
                if i == n_passes:
                    x_batch = np.zeros((current_batch, x.shape[1]))
                    current_batch = len(idx_train) - i * batch_size
                for j in range(current_batch):
                    x_batch[j] = x[idx_train[i * batch_size + j]]
                self.scaler.partial_fit(x_batch)
            logger.info(f'Fitting scaler done, took {time() - s} seconds')

            # Setup the overall dataset and the individual train and eval subsets
            ds = H5Dataset(x_train, y_train)
            logger.info(f'Creating training dataset')
            train_data = torch.utils.data.Subset(ds, idx_train)

            logger.info(f'Creating validation dataset')
            val_data = torch.utils.data.Subset(ds, idx_val)

            # Setup the data loaders
            logger.info(f'Creating training and validation dataloaders with batch size {self.args["minibatch_size"]} and {self.num_workers} workers')
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.args["minibatch_size"], num_workers=self.num_workers, pin_memory=False if self.device == torch.device('cpu') else True)
            val_dataloader = DataLoader(val_data, batch_size=self.args["minibatch_size"], num_workers=self.num_workers, pin_memory=False if self.device == torch.device('cpu') else True)
        else:
            self.scaler.fit(x_train)
            train_data = TensorDataset(
                torch.from_numpy(x_train),
                torch.from_numpy(y_train),
            )
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.args["minibatch_size"], num_workers=self.num_workers)
            val_dataloader = None

        # Remember to close H5 file
        f.close()

        best_loss = np.inf
        best_model = None
        stall_count = 0
        stop = False
        logger.info(f'[{self.name}] Training model')
        for epoch in tqdm(range(self.args["max_epochs"]), desc="epochs", leave=True):
            if epoch > 0 and stop:
                break
            model.train()

            # loop over the training set
            total_train_loss = 0
            steps = 0
            for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"epoch: {epoch}", leave=False):
                x, y = data

                # scale x
                x = torch.from_numpy(self.scaler.transform(x.numpy())).to(self.device)

                y_hat = model(x)
                if num_outputs == 1:
                    loss = loss_function(y_hat.sigmoid(), y.to(self.device))
                else:
                    yhat = torch.nn.functional.softmax(y_hat, dim=1)
                    loss = loss_function(yhat, y.to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss
                steps += 1

            total_train_loss = total_train_loss.cpu().detach().numpy() / steps

            if val_dataloader is not None:
                total_val_loss = 0
                # switch off autograd for evaluation
                with torch.no_grad():
                    # set the model in evaluation mode
                    model.eval()
                    for i, data in enumerate(val_dataloader):
                        x, y = data

                        # scale x
                        x = torch.from_numpy(self.scaler.transform(x.numpy())).to(self.device)

                        y_hat = model(x)
                        if num_outputs == 1:
                            total_val_loss += loss_function(y_hat.sigmoid(), y.to(self.device))
                        else:
                            yhat = torch.nn.functional.softmax(y_hat, dim=1)
                            total_val_loss += loss_function(yhat, y.to(self.device))
                total_val_loss = total_val_loss.cpu().detach().numpy() / steps
                scheduler.step(total_val_loss)

                if total_val_loss >= best_loss:
                    stall_count += 1
                    if stall_count >= self.args["patience"]:
                        stop = True
                        print(f"\n<Stopped at Epoch {epoch + 1}>")
                else:
                    best_loss = total_val_loss
                    best_model = copy.deepcopy(model)
                    if not stop:
                        stall_count = 0
            else:
                scheduler.step(total_train_loss)
                if total_train_loss >= best_loss:
                    stall_count += 1
                    if stall_count >= self.args["patience"]:
                        stop = True
                        print(f"\n<Stopped at Epoch {epoch + 1}>")
                else:
                    best_loss = total_train_loss
                    best_model = copy.deepcopy(model)
                    if not stop:
                        stall_count = 0

        logger.info(f'[{self.name}] Training complete')
        self.model = best_model
        return self.model

    def predict(self, x, output_probs=False):
        x = self.scaler.transform(x)
        with torch.no_grad():
            # set the model in evaluation mode
            self.model.eval()

            output = self.model(torch.from_numpy(x).to(self.device))

            if output_probs:
                probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()

            if self.num_classes > 2:
                yhat = self.classes[np.argmax(output.cpu().detach().numpy(), axis=1)]
            else:
                yhat = torch.sigmoid(output)
                yhat = np.round(yhat.cpu().detach().numpy())

            return (yhat, probs) if output_probs else yhat
