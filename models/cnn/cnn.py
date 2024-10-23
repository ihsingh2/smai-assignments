""" Provides the CNN class. """

from collections import deque
from typing import List, Literal, Self

import numpy as np
import numpy.typing as npt
import torch
import wandb
from torchinfo import ModelStatistics, summary


# pylint: disable-next=too-many-instance-attributes
class CNN:
    """ Implements Convolutional Neural Network, using VGG-Net Architecture with Dropout.

    The layer dimensions are set according to the Multi MNIST (double_mnist) dataset.
        Input:  (n, 1, 128, 128)
        Output: (n, 1) for regression
                (n, 4) for single-label-classification
                (n, 10) for multi-label-classification
    """

    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def __init__(
        self, activation: Literal['relu', 'sigmoid', 'tanh'], pool: Literal['avgpool', 'maxpool'],
        task: Literal['regression', 'single-label-classification', 'multi-label-classification'],
        optimizer: Literal['sgd', 'adam'], lr: float = 1e-4, num_blocks: int = 5,
        kernel_size: int = 3, dropout: float = 0, num_epochs: int = 10, batch_size: int = 32,
        random_seed: int | None = 0
    ):
        """ Initializes the model hyperparameters.

        Args:
            activation: The activation function to use for non-linearity.
            pool: The pooling layer to use after each block of convolution.
            task: Indicator for which task the model should perform.
                  The final output is transformed accordingly.
            optimizer: The optimizer to use for weight updates.
            lr: The learning rate for weight updates.
            num_blocks: The number of blocks of convolution layers to add to the network.
            dropout: The dropout rate to use after each block of convolution layer and
                     each fully connected layer.
            num_epochs: Number of iterations over the training dataset.
            batch_size: Number of samples from the training dataset to process in a batch.
        """

        # Validate the passed arguments
        assert activation in ['relu', 'sigmoid', 'tanh'], \
                                            f'Got unrecognized value for activation {activation}'
        assert pool in ['avgpool', 'maxpool'], f'Got unrecognized value for pool {pool}'
        assert task in \
            ['regression', 'single-label-classification', 'multi-label-classification'], \
                                                        f'Got unrecognized value for task {task}'
        assert optimizer in ['sgd', 'adam'], f'Got unrecognized value for optimizer {optimizer}'
        assert lr > 0, 'lr should be positive'
        assert num_blocks >= 0, 'num_blocks should be non-negative'
        assert kernel_size > 0, 'kernel_size should be positive'
        assert dropout >= 0, 'momentum should be non-negative'
        assert num_epochs > 0, 'num_epochs should be positive'
        assert batch_size > 0, 'batch_size should be positive'

        # Store the passed arguments
        self.task = task
        self.num_blocks = num_blocks
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Reinitialize the random number generator
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # Initialize the model parameters
        self.network = self._init_network(num_blocks, kernel_size, dropout, activation, pool)
        self.optimizer = self._get_optimizer(optimizer, lr)
        self.loss_function = self._get_loss(task)
        self.val_losses = None


    def cpu(self) -> Self:
        """ Moves the model to CPU. """

        self.network = self.network.cpu()
        return self


    def cuda(self) -> Self:
        """ Moves the model to GPU, if available. """

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.network = self.network.to(device)
        return self


    def _early_stopping(self, val_loss: float) -> bool:
        """ Checks the suitability for early stopping of gradient descent. """

        self.val_losses.append(val_loss)

        # If sufficient samples for loss available for comparision
        if len(self.val_losses) == self.val_losses.maxlen:

            # Compute mean of first half and second half
            midpoint = self.val_losses.maxlen // 2
            previous_loss_pattern = np.mean(list(self.val_losses)[:midpoint])
            current_loss_pattern = np.mean(list(self.val_losses)[midpoint:])

            # If second half loss is greater, stop
            if current_loss_pattern > previous_loss_pattern:
                return True

        return False


    def eval(self) -> Self:
        """ Sets the model to eval state. """

        self.network = self.network.eval()
        return self


    def feature_maps(self, X: torch.Tensor) -> List[npt.NDArray[float]]:
        """ Extracts the feature maps for a given input. """

        with torch.no_grad():
            maps = []
            for idx in range(self.num_blocks):
                maps.append(self.forward(X, idx).detach().numpy())
            return maps


    def fit(self, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, \
                                        verbose: bool = False, wandb_log: bool = False) -> Self:
        """ Fits the model for given training data, using validation data for early stopping. """

        # Load the model to GPU
        self.cuda()

        # Dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size
        )

        # Queue to store recent val losses for early stopping
        self.val_losses = deque(maxlen=4)

        # Iterate over the datasets
        for epoch in range(self.num_epochs):

            # Train
            self.train()
            train_loss = self.pass_epoch(train_dataloader, train=True)

            # Validate
            self.eval()
            val_loss = self.pass_epoch(val_dataloader)

            # Log metrics
            if verbose:
                print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
            if wandb_log:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })

            # Early stopping
            if self._early_stopping(val_loss):
                break

        # Unload the model from GPU
        self.cpu()

        return self


    def forward(self, X: torch.Tensor, index: int | None = None) -> torch.Tensor:
        """ Performs forward propagation on the input image and returns the output of the
        final layer (or an intermediate layer optionally). """

        if index is None:
            return self.network.forward(X)

        if not 0 <= index < self.num_blocks + 3:
            raise ValueError(f'index should be in range [0, {self.num_blocks + 2}]')

        for idx, layer in enumerate(self.network):
            X = layer(X)
            if idx == index:
                break

        return X


    def _get_activation(self, activation: Literal['relu', 'sigmoid', 'tanh']) -> torch.nn.Module:
        """ Returns the activation function, specified by the description. """

        if activation == 'relu':
            return torch.nn.ReLU()

        if activation == 'sigmoid':
            return torch.nn.Sigmoid()

        return torch.nn.Tanh()


    def _get_device(self) -> torch.device:
        """ Returns the device the network is stored on. """

        return next(self.network.parameters()).device


    def _get_loss(self, task: Literal['regression', 'single-label-classification', \
                                                'multi-label-classification']) -> torch.nn.Module:
        """ Returns the loss function, for the specified task. """

        if task == 'single-label-classification':
            return torch.nn.CrossEntropyLoss()

        if task == 'multi-label-classification':
            return torch.nn.BCELoss()

        return torch.nn.MSELoss()


    def _get_optimizer(self, optimizer: Literal['sgd', 'adam'], lr: float) \
                                                                        -> torch.optim.Optimizer:
        """ Returns the optimizer, specified by the description. """

        if optimizer == 'sgd':
            return torch.optim.SGD(self.network.parameters(), lr=lr)

        return torch.optim.Adam(self.network.parameters(), lr=lr)


    def _get_pool(self, pool: Literal['avgpool', 'maxpool']) -> torch.nn.Module:
        """ Returns the pooling layer, specified by the description. """

        if pool == 'avgpool':
            return torch.nn.AvgPool2d(kernel_size=2, stride=2)

        return torch.nn.MaxPool2d(kernel_size=2, stride=2)


    def _get_summary(self) -> ModelStatistics:
        """ Prints a summary of the network. """

        return summary(
            model=self.network,
            input_size=(1, 1, 128, 128),
            col_names=["input_size", "output_size", "num_params"],
            col_width=25
        )


    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def _init_network(
        self, num_blocks: int, kernel_size: int, dropout: float,
        activation: Literal['relu', 'sigmoid', 'tanh'], pool: Literal['avgpool', 'maxpool']
    ) -> torch.nn.Module:
        """ Initializes the network, with reference to VGG-Net Architecture with Dropout. """

        layers = []

        # Convolutional blocks
        for idx in range(num_blocks):
            layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(max(64*(idx), 1), 64*(idx+1), \
                                                        kernel_size=kernel_size, padding='same'),
                self._get_activation(activation),
                torch.nn.Conv2d(64*(idx+1), 64*(idx+1), kernel_size=kernel_size, padding='same'),
                self._get_activation(activation),
                self._get_pool(pool),
                torch.nn.Dropout(dropout)
            ))

        # Fully connected block 1
        layers.append(torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(max(64 * num_blocks, 1) * ((2 ** (7 - num_blocks)) ** 2), 512),
            self._get_activation(activation),
            torch.nn.Dropout(dropout)
        ))

        # Fully connected block 2
        layers.append(torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            self._get_activation(activation),
            torch.nn.Dropout(dropout)
        ))

        # Output layer
        if self.task == 'single-label-classification':
            layers.append(torch.nn.Sequential(
                torch.nn.Linear(128, 4),
                torch.nn.Softmax(dim=1)
            ))
        elif self.task == 'multi-label-classification':
            layers.append(torch.nn.Sequential(
                torch.nn.Linear(128, 10),
                torch.nn.Sigmoid()
            ))
        else:
            layers.append(torch.nn.Linear(128, 1))

        return torch.nn.Sequential(*layers)


    def pass_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool = False) -> float:
        """ Iterates over the dataset, optionally trains the model and returns the total loss. """

        # Device the model is stored on
        device = self._get_device()

        # Total loss
        loss = 0

        for _, (x, y) in enumerate(dataloader):

            # Compute predictions
            x = x.to(device)
            y = y.to(device)
            y_pred = self.forward(x)

            # Ensure compatibility with cross entropy module
            if self.task == 'single-label-classification':
                y = y.squeeze(1).long()

            # Compute loss
            loss_batch = self.loss_function(y_pred, y)
            loss += loss_batch.detach().cpu().item()

            # Update step
            if train:
                loss_batch.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Average loss
        loss /= len(dataloader)

        return loss


    def predict(self, X: torch.Tensor | torch.utils.data.Dataset) -> torch.Tensor:
        """ Returns the model output for an image, after post-processing based on task. """

        # Inference mode
        self.eval()
        with torch.inference_mode():

            # Compute prediction on CPU - The tensor may be too large to store on VRAM
            if isinstance(X, torch.Tensor):
                y = self.forward(X)

            # Compute prediction on GPU - Setup dataloader with configured batch size
            elif isinstance(X, torch.utils.data.Dataset):
                self.cuda()
                device = self._get_device()

                y = []
                dataloader = torch.utils.data.DataLoader(X, batch_size=self.batch_size)

                for _, (x, _) in enumerate(dataloader):
                    x = x.to(device)
                    y_pred = self.forward(x).cpu()
                    y.append(y_pred)

                y = torch.cat(y, dim=0)
                self.cpu()

            else:
                raise ValueError('Expected a Tensor or Dataset')

        # Clip the unbounded regression output
        if self.task == 'regression':
            y = y.clip(0, 3).round()

        # Predict the class with largest logit
        elif self.task == 'single-label-classification':
            y = y.argmax(axis=1).unsqueeze(1)

        # Binary thresholding on each class probability
        elif self.task == 'multi-label-classification':
            y = torch.where(y > 0.5, 1, 0)

        return y


    def train(self) -> Self:
        """ Sets the model to train state. """

        self.network = self.network.train()
        return self
