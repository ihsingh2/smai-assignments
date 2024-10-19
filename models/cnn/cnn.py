""" Provides the CNN class. """

from typing import Literal

import torch
from torchinfo import ModelStatistics, summary


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
        optimizer: Literal['sgd', 'adam'], lr: float = 1e-4, momentum: float = 0,
        weight_decay: float = 0, num_blocks: int = 5, kernel_size: int = 3,
        dropout: float = 0, num_epochs: int = 10, batch_size: int = 32
    ):
        """ Initializes the model hyperparameters.

        Args:
            task: Indicator for which task the model should perform.
                  The final output is transformed accordingly.
            activation: The activation function to use for non-linearity.
            optimizer: The optimizer to use for weight updates.
            lr: The learning rate for weight updates.
            momentum: The momentum factor to use with optimizer (if supported).
            weight_decay: The weight decay coefficient to use with optimizer (if supported).
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
        assert momentum >= 0, 'momentum should be non-negative'
        assert weight_decay >= 0, 'weight_decay should be non-negative'
        assert num_blocks > 0, 'num_blocks should be positive'
        assert kernel_size > 0, 'kernel_size should be positive'
        assert dropout >= 0, 'momentum should be non-negative'
        assert num_epochs > 0, 'num_epochs should be positive'
        assert batch_size > 0, 'batch_size should be positive'

        # Store the passed arguments
        self.task = task
        self.num_blocks = num_blocks
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Initialize the model parameters
        self.network = self._init_network(num_blocks, kernel_size, dropout, activation, pool)
        self.loss_function = self._get_loss(task)
        self.optimizer = self._get_optimizer(optimizer, lr, momentum, weight_decay)


    def fit(self, dataset: torch.utils.data.Dataset):
        """ Fits the model for the given training data. """

        # TODO validation

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        for epoch in range(self.num_epochs):
            loss = self.pass_epoch(dataloader, train=True)
            print(epoch, loss)


    def forward(self, X: torch.Tensor, index: int | None = None):
        """ Performs forward propagation on the input image and returns the output of the
        final layer (or an intermediate layer optionally). """

        if index is None:
            return self.network.forward(X)

        if not 0 <= index < self.num_blocks + 3:
            raise ValueError(f'index should be in range(0, {self.num_blocks + 2})')

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


    def _get_optimizer(self, optimizer: Literal['sgd', 'adam'], lr: float, momentum: float, \
                                                    weight_decay: float) -> torch.optim.Optimizer:
        """ Returns the optimizer, specified by the description. """

        if optimizer == 'sgd':
            return torch.optim.SGD(self.network.parameters(), lr=lr, momentum=momentum, \
                                                                        weight_decay=weight_decay)

        return torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)


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
            torch.nn.Linear(64 * num_blocks * ((2 ** (7 - num_blocks)) ** 2), 512),
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

        # Device to run model on
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        return torch.nn.Sequential(*layers).to(device)


    def pass_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool = False) -> float:
        """ Performs one pass over the dataset, optionally trains the model
        and returns the total loss. """

        if train:
            self.network.train()
        else:
            self.network.eval()

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
            loss += loss_batch.detach().cpu()

            # Update step
            if train:
                loss_batch.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        return loss
