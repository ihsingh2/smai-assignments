""" Provides the CNNAutoEncoder class. """

from typing import Literal, Self, Tuple

import torch
from torchinfo import ModelStatistics, summary


# pylint: disable-next=too-many-instance-attributes
class CNNAutoEncoder:
    """ Implements CNN based AutoEncoder, using VGG-Net Architecture with Dropout.

    The layer dimensions are set according to the Fashion MNIST dataset.
        Input:  (n, 1, 28, 28)
        Output: (n, latent_dimension)
    """


    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def __init__(
        self, latent_dimension: int, activation: Literal['relu', 'sigmoid', 'tanh'],
        pool: Literal['avgpool', 'maxpool'], optimizer: Literal['sgd', 'adam'], lr: float = 1e-4,
        num_blocks: int = 5, kernel_size: int = 3, dropout: float = 0, num_epochs: int = 10,
        batch_size: int = 32
    ):
        """ Initializes the model hyperparameters.

        Args:
            activation: The activation function to use for non-linearity.
            pool: The pooling layer to use after each block of convolution.
            optimizer: The optimizer to use for weight updates.
            lr: The learning rate for weight updates.
            num_blocks: The number of blocks of convolution layers to add to the network.
            dropout: The dropout rate to use after each block of convolution layer and
                     each fully connected layer.
            num_epochs: Number of iterations over the training dataset.
            batch_size: Number of samples from the training dataset to process in a batch.
        """

        # Validate the passed arguments
        assert latent_dimension > 0, 'latent_dimension should be positive'
        assert activation in ['relu', 'sigmoid', 'tanh'], \
                                            f'Got unrecognized value for activation {activation}'
        assert pool in ['avgpool', 'maxpool'], f'Got unrecognized value for pool {pool}'
        assert optimizer in ['sgd', 'adam'], f'Got unrecognized value for optimizer {optimizer}'
        assert lr > 0, 'lr should be positive'
        assert num_blocks > 0, 'num_blocks should be positive'
        assert kernel_size > 0, 'kernel_size should be positive'
        assert dropout >= 0, 'momentum should be non-negative'
        assert num_epochs > 0, 'num_epochs should be positive'
        assert batch_size > 0, 'batch_size should be positive'

        # Store the passed arguments
        self.latent_dimension = latent_dimension
        self.num_blocks = num_blocks
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Initialize the model parameters
        self.encoder = self._init_encoder(num_blocks, kernel_size, dropout, activation, pool)
        self.decoder = self._init_decoder(num_blocks, kernel_size, dropout, activation, pool)
        self.optimizer = self._get_optimizer(optimizer, lr)
        self.loss_function = torch.nn.MSELoss()


    def cpu(self) -> Self:
        """ Moves the encoder, decoder models to CPU. """

        self.encoder = self.encoder.cpu()
        self.decoder = self.decoder.cpu()
        return self


    def cuda(self) -> Self:
        """ Moves the encoder, decoder models to GPU, if available. """

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        return self


    def decode(self, X: torch.Tensor) -> torch.Tensor:
        """ Reconstructs the image using the compressed latent space vector. """

        return self.decoder.forward(X)


    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """ Reduces the spatial dimension of the input image."""

        return self.encoder.forward(X)


    def eval(self) -> Self:
        """ Sets the encoder, decoder models to eval state. """

        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()
        return self

    # pylint: disable=duplicate-code

    def fit(self, dataset: torch.utils.data.Dataset) -> Self:
        """ Fits the model for the given training data. """

        # Set the models to train
        self.cuda()
        self.train()

        # Dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Iterate over the dataset
        for epoch in range(self.num_epochs):
            loss = self.pass_epoch(dataloader)
            print(f'Epoch {epoch}, Loss: {loss}')

        # Set the models to eval
        self.cpu()
        self.eval()

        return self

    # pylint: enable=duplicate-code

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Compresses the input image and reconstructs it back. """

        return self.decode(self.encode(X))


    def _get_activation(self, activation: Literal['relu', 'sigmoid', 'tanh']) -> torch.nn.Module:
        """ Returns the activation function, specified by the description. """

        if activation == 'relu':
            return torch.nn.ReLU()

        if activation == 'sigmoid':
            return torch.nn.Sigmoid()

        return torch.nn.Tanh()


    def _get_device(self) -> torch.device:
        """ Returns the device the network is stored on. """

        assert next(self.encoder.parameters()).device == next(self.decoder.parameters()).device, \
                                            'Expected encoder and decoder to be on the same device'
        return next(self.encoder.parameters()).device


    def _get_optimizer(self, optimizer: Literal['sgd', 'adam'], lr: float) \
                                                                        -> torch.optim.Optimizer:
        """ Returns the optimizer, specified by the description. """

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())

        if optimizer == 'sgd':
            return torch.optim.SGD(params, lr=lr)

        return torch.optim.Adam(params, lr=lr)


    def _get_pool(self, pool: Literal['avgpool', 'maxpool']) -> torch.nn.Module:
        """ Returns the pooling layer, specified by the description. """

        if pool == 'avgpool':
            return torch.nn.AvgPool2d(kernel_size=2, stride=2)

        return torch.nn.MaxPool2d(kernel_size=2, stride=2)


    def _get_summary(self) -> ModelStatistics:
        """ Prints a summary of the network. """

        encoder_summary = summary(
            model=self.encoder,
            input_size=(1, 1, 28, 28),
            col_names=["input_size", "output_size", "num_params"],
            col_width=25
        )
        decoder_summary = summary(
            model=self.decoder,
            input_size=(1, self.latent_dimension),
            col_names=["input_size", "output_size", "num_params"],
            col_width=25
        )
        return encoder_summary, decoder_summary


    def _get_upsample(self, output_size: Tuple[int, int], pool: Literal['avgpool', 'maxpool']) \
                                                                                -> torch.nn.Module:
        """ Returns the upsampling layer for unpooling, analogous to the pooling description. """

        if pool == 'avgpool':
            return torch.nn.Upsample(size=output_size, mode='bilinear')

        return torch.nn.Upsample(size=output_size, mode='nearest')


    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def _init_decoder(
        self, num_blocks: int, kernel_size: int, dropout: float,
        activation: Literal['relu', 'sigmoid', 'tanh'], pool: Literal['avgpool', 'maxpool']
    ) -> torch.nn.Module:
        """ Initializes the decoder, with reference to VGG-Net Architecture with Dropout. """

        layers = []

        # Size of layerwise activation map
        map_size = [28, 14, 7, 3, 1]

        # Output layer
        layers.append(torch.nn.Linear(self.latent_dimension, 128))

        # Fully connected block 2
        layers.append(torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            self._get_activation(activation),
            torch.nn.Linear(128, 512),
        ))

        # Fully connected block 1
        layers.append(torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            self._get_activation(activation),
            torch.nn.Linear(512, max(64 * num_blocks, 1) * (map_size[num_blocks] ** 2)),
            torch.nn.Unflatten(1, (64 * num_blocks, map_size[num_blocks], map_size[num_blocks])),
        ))

        # Convolutional blocks
        for idx in reversed(range(num_blocks)):
            layers.append(torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                self._get_upsample((map_size[idx], map_size[idx]), pool),
                self._get_activation(activation),
                torch.nn.ConvTranspose2d(64*(idx+1), 64*(idx+1), kernel_size=kernel_size, \
                                                                        padding=kernel_size // 2),
                self._get_activation(activation),
                torch.nn.ConvTranspose2d(64*(idx+1), max(64*(idx), 1), kernel_size=kernel_size, \
                                                                        padding=kernel_size // 2)
            ))

        return torch.nn.Sequential(*layers)


    # pylint: disable=duplicate-code

    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def _init_encoder(
        self, num_blocks: int, kernel_size: int, dropout: float,
        activation: Literal['relu', 'sigmoid', 'tanh'], pool: Literal['avgpool', 'maxpool']
    ) -> torch.nn.Module:
        """ Initializes the encoder, with reference to VGG-Net Architecture with Dropout. """

        layers = []

        # Validate the argument
        assert 0 <= num_blocks <= 4, 'Number of blocks out of range for the image dimension'

        # Size of layerwise activation map
        map_size = [28, 14, 7, 3, 1]

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
            torch.nn.Linear(max(64 * num_blocks, 1) * (map_size[num_blocks] ** 2), 512),
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
        layers.append(torch.nn.Linear(128, self.latent_dimension))

        return torch.nn.Sequential(*layers)

    # pylint: enable=duplicate-code

    def pass_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """ Performs one pass over the dataset, optionally trains the model
        and returns the total loss. """

        # Device the model is stored on
        device = self._get_device()

        # Total loss
        loss = 0

        for _, x in enumerate(dataloader):

            # Compute predictions
            x = x.to(device)
            x_pred = self.forward(x)

            # Compute loss
            loss_batch = self.loss_function(x_pred, x)
            loss += loss_batch.detach().cpu()

            # Update step
            loss_batch.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Average loss
        loss /= len(dataloader)

        return loss


    def train(self) -> Self:
        """ Sets the encoder, decoder models to train state. """

        self.encoder = self.encoder.train()
        self.decoder = self.decoder.train()
        return self
