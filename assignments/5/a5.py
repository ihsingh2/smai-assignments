""" Assignment 5: Kernel Density Estimation, Hidden Markov Model and Recurrent Neural Network. """

import sys
from typing import List, Tuple

import librosa
import matplotlib.pyplot as plt
import nltk
import numpy as np
import numpy.typing as npt
import torch
from hmmlearn import hmm
from PIL import Image, ImageDraw, ImageFont

# pylint: disable=wrong-import-position

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

from models.kde import KDE
from models.gmm import GMM

# pylint: enable=wrong-import-position


def bit_counting() -> None:
    """ Applies RNN on a synthetic binary dataset for counting ones in a bit stream. """

    def generate_random_lengths(num_seq: int, max_len: int) -> List[int]:
        """ Generates random lengths, upto a maximum length. """

        length = []
        for _ in range(num_seq):
            length.append(np.random.randint(1, max_len + 1))
        return length

    def generate_binary_sequences(num_seq: int, max_len: int, constant_length: bool = False) \
                                                                -> Tuple[List[List], List[int]]:
        """ Generates binary sequences of varying length, upto a maximum length. """

        list_count = []
        list_sequence = []
        list_length = generate_random_lengths(num_seq, max_len)

        for length in list_length:
            if constant_length:
                sequence = np.random.choice([0, 1], size=max_len)
            else:
                sequence = np.random.choice([0, 1], size=length)

            list_count.append(int(np.sum(sequence)))
            list_sequence.append(sequence.tolist())

        if constant_length:
            return np.array(list_sequence), np.array(list_count)

        return np.array(list_sequence, dtype=object), np.array(list_count)

    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) \
                                                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Pad sequences to the maximum length in the batch. """

        sequences, counts = zip(*batch)
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, \
                                                                                padding_value=0)
        counts = torch.stack(counts)
        return padded_sequences.unsqueeze(-1), counts, lengths

    class BitCountDataset(torch.utils.data.Dataset):
        """ Dataset for binary sequences and count of ones. """

        def __init__(self, sequence: npt.NDArray[int], length: npt.NDArray[int]):
            self.sequence = sequence
            self.length = length

        def __len__(self):
            return len(self.sequence)

        def __getitem__(self, idx):
            return torch.tensor(self.sequence[idx], dtype=torch.float32), \
                                                torch.tensor(self.length[idx], dtype=torch.float32)

    class BitCounterRNN(torch.nn.Module):
        """ RNN for counting the number of ones in binary sequences. """

        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.rnn = torch.nn.RNN(1, hidden_size, batch_first=True)
            self.fc = torch.nn.Linear(hidden_size, 1)

        # pylint: disable-next=missing-function-docstring
        def forward(self, sequence, length):
            sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence, length.cpu(), \
                                                        batch_first=True, enforce_sorted=False)
            _, hidden = self.rnn(sequence)
            output = self.fc(hidden[-1])
            return output.squeeze()

    # Log function call
    print('--- bit_counting')

    # Generate synthetic dataset
    X, y = generate_binary_sequences(100000, 16)
    print('Generated 100000 sequences')
    print('Example sequences:')
    for idx in range(3):
        print(X[idx], y[idx])

    # Train val test split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataset = BitCountDataset(X_train, y_train)
    val_dataset = BitCountDataset(X_val, y_val)
    test_dataset = BitCountDataset(X_test, y_test)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, \
                                                                            collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, \
                                                                            collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, \
                                                                            collate_fn=collate_fn)

    # Model and optimizer
    model = BitCounterRNN(hidden_size=32).to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Iterate over dataset
    for epoch in range(10):

        # Train
        model.train()
        train_loss = 0
        for sequences, labels, lengths in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels, lengths in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)
                outputs = model(sequences, lengths)
                val_loss += criterion(outputs, labels).item()

        # Log loss
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}, Train MAE: {train_loss}, Validation MAE: {val_loss}')

    # Test
    model.eval()
    test_loss = 0
    random_loss = 0
    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            outputs = model(sequences, lengths)
            random_outputs = torch.Tensor(generate_random_lengths(len(sequences), 16)).to(device)
            test_loss += criterion(outputs, labels).item()
            random_loss += criterion(random_outputs, labels).item()

    # Log loss
    test_loss /= len(test_loader)
    random_loss /= len(test_loader)
    print(f'Test MAE: {test_loss}')
    print(f'Random MAE: {random_loss}')

    # Test generalization with sequence length
    sequence_test_loss = []
    for idx in range(1, 33):
        X_test, y_test = generate_binary_sequences(10000, idx, constant_length=True)
        test_dataset = BitCountDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, \
                                                                            collate_fn=collate_fn)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for sequences, labels, lengths in test_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)
                outputs = model(sequences, lengths)
                test_loss += criterion(outputs, labels).item()

        test_loss /= len(test_loader)
        sequence_test_loss.append(test_loss)
        print(f'Length {idx}, MAE: {test_loss}')

    # Plot loss versus sequence length
    output_path = 'figures/rnn_bit_counting_generalization.png'
    plt.figure(figsize=(14, 10))
    plt.plot(range(1, 33), sequence_test_loss)
    plt.title('MAE for Different Sequence Lengths')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Sequence Length')
    plt.xticks(range(1, 33))
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)
    print()


def kernel_density_estimation() -> None:
    """ Applies KDE on a synthetic dataset and contrasts it with GMM. """

    def sample_points_in_circle(n: int, r: float, c_x: float, c_y: float) -> npt.NDArray[float]:
        """ Samples n points in a circle of radius r, centered at (c_x, c_y). """

        points = []
        for _ in range(n):
            radius = r * np.sqrt(np.random.uniform(0, 1))
            angle = np.random.uniform(0, 2 * np.pi)
            x = c_x + radius * np.cos(angle)
            y = c_y + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)

    # Log function call
    print('--- kernel_density_estimation')

    # Generate synthetic dataset
    points = np.vstack(( \
        sample_points_in_circle(3000, 2, 0, 0),
        sample_points_in_circle(500, 0.25, 1, 1)
    ))
    points += 0.01 * np.random.randn(3500, 2)

    # Plot synthetic dataset
    output_path = 'figures/synthetic_data.png'
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=1)
    plt.title('Synthetic Dataset')
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)

    # Plot KDE estimate
    kde = KDE('gaussian', 0.25).fit(points)
    output_path = 'figures/synthetic_data_kde.png'
    kde.visualize(output_path)
    print(output_path)

    # Plot GMM memberships for two components
    output_path = 'figures/synthetic_data_gmm_2.png'
    gmm = GMM(2).fit(points)
    gmm.visualize(output_path)
    print(output_path)

    # Plot GMM memberships for three components
    output_path = 'figures/synthetic_data_gmm_3.png'
    gmm = GMM(3).fit(points)
    gmm.visualize(output_path)
    print(output_path)

    # Plot GMM memberships for four components
    output_path = 'figures/synthetic_data_gmm_4.png'
    gmm = GMM(4).fit(points)
    gmm.visualize(output_path)
    print(output_path)

    # Plot GMM memberships for five components
    output_path = 'figures/synthetic_data_gmm_5.png'
    gmm = GMM(5).fit(points)
    gmm.visualize(output_path)
    print(output_path)

    print()


def optimal_character_recognition() -> None:
    """ Applies RNN on a NLTK words dataset rendered onto images for text recognition. """

    def render_text(word: str) -> npt.NDArray[np.uint8]:
        """ Render text onto an image and returns the image. """

        # Image
        image = Image.new("L", (256, 64), 255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('LiberationMono-Regular.ttf', 16)

        # Bounding box for centering
        text_bbox = draw.textbbox((0, 0), word, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw text
        text_x = (image.width - text_width) // 2
        text_y = (image.height - text_height) // 2
        draw.text((text_x, text_y), word, font=font, fill=0)

        return np.array(image)

    def encode_character(char: str) -> int:
        """ Maps lowercase ASCII characters to [1, 26] range. """

        return ord(char) - ord('a') + 1

    def encode_string(characters: str) -> npt.NDArray:
        """ Encodes a sequence of lowercase ASCII characters. """

        return [encode_character(char) for char in characters]

    def decode_character(index: int) -> str:
        """ Remaps [1, 26] range to lowercase ASCII characters. """

        return chr(index + ord('a') - 1)

    def decode_string(indices: npt.NDArray) -> str:
        """ Decodes a sequence of encoded characters. """

        return "".join([decode_character(idx) for idx in indices])

    def decode_ctc_output(output: npt.NDArray) -> npt.NDArray:
        """ Merges repeated characters and removes blank (i.e. 0) tokens. """

        decoded_output = []
        prev = -1
        for char in output:
            if char not in (prev, 0):
                decoded_output.append(char)
            prev = char
        return np.array(decoded_output)

    def collate_fn_ctc(batch: List[Tuple[torch.Tensor, torch.Tensor]]) \
                                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Concatenates sequences in a batch and stores the original lengths. """

        images, labels = zip(*batch)
        images = torch.stack(images, 0)
        label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
        labels_concat = torch.cat(labels)
        output_lengths = torch.full((len(batch),), 16, dtype=torch.long)
        return images, labels_concat, label_lengths, output_lengths

    class WordImageDataset(torch.utils.data.Dataset):
        """ Dataset for word renderings. """

        def __init__(self, images: npt.NDArray[np.uint8], words: npt.NDArray[object]):
            self.images = images
            self.words = words

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return torch.tensor(self.images[idx], dtype=torch.float32), \
                                                torch.tensor(self.words[idx], dtype=torch.long)

    class ConvBlock(torch.nn.Module):
        """ Basic CNN block, including BatchNorm, non-linearity and Maxpool. """

        def __init__(self, input_channels: int, output_channels: int):
            super().__init__()
            self.sequential = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding='same'),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2, 2),
            )

        # pylint: disable-next=missing-function-docstring
        def forward(self, x):
            return self.sequential(x)

    class CharacterRecognizer(torch.nn.Module):
        """ Encoder decoder model for recognizing characters in an image. """

        def __init__(self):
            super().__init__()
            self.cnn = torch.nn.Sequential(
                ConvBlock(1, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
                ConvBlock(256, 512),
                torch.nn.AdaptiveMaxPool2d((1, 16))
            )
            self.rnn = torch.nn.RNN(512, 512, 2, bidirectional=True, dropout=0.2)
            self.fc = torch.nn.Linear(1024, 27)
            self.softmax = torch.nn.LogSoftmax(dim=2)

        # pylint: disable-next=missing-function-docstring
        def forward(self, x):
            x = self.cnn(x).squeeze(2).permute(2, 0, 1)
            x, _ = self.rnn(x)
            x = self.fc(x)
            x = self.softmax(x)
            return x

    # Log function call
    print('--- optimal_character_recognition')

    # Download words
    nltk.download('words')
    words_list = nltk.corpus.words.words()
    words_list = [ word.lower() for word in words_list[:100000] ]

    # Rendered words on images
    words_images = np.array([ render_text(word) for word in words_list ])

    # Data
    X = 255 - words_images.reshape((-1, 1, 64, 256))
    y = np.array([ encode_string(word) for word in words_list ], dtype=object)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # Dataset
    train_dataset = WordImageDataset(X_train, y_train)
    val_dataset = WordImageDataset(X_val, y_val)
    test_dataset = WordImageDataset(X_test, y_test)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, \
                                                                        collate_fn=collate_fn_ctc)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, \
                                                                        collate_fn=collate_fn_ctc)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, \
                                                                        collate_fn=collate_fn_ctc)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and optimizer
    model = CharacterRecognizer().to(device)
    criterion = torch.nn.CTCLoss(zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    # Iterate over dataset
    for epoch in range(5):

        # Train
        model.train()
        train_loss = 0.0
        for images, labels, label_lengths, input_lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            input_lengths = input_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, label_lengths, input_lengths in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                label_lengths = label_lengths.to(device)
                input_lengths = input_lengths.to(device)

                log_probs = model(images)
                loss = criterion(log_probs, labels, input_lengths, label_lengths)
                val_loss += loss.item()

        # Log loss
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

        # Examples
        with torch.no_grad():
            for images, labels, label_lengths, input_lengths in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                label_lengths = label_lengths.to(device)
                input_lengths = input_lengths.to(device)

                log_probs = model(images)
                preds = torch.argmax(log_probs, dim=2).permute(1, 0)

                cumulative_lengths = torch.cumsum(label_lengths, dim=0)
                start_indices = cumulative_lengths - label_lengths
                end_indices = cumulative_lengths

                print('Example predictions:')
                for i in range(3):
                    target = labels[start_indices[i].item(): end_indices[i].item()].cpu().numpy()
                    pred = decode_ctc_output(preds[i].cpu().numpy())
                    print(f'Actual: {decode_string(target)}, Prediction: {decode_string(pred)}')

                break

    # Test
    model.eval()
    total_correct = 0
    random_correct = 0
    total_chars = 0
    with torch.no_grad():
        for images, labels, label_lengths, input_lengths in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            input_lengths = input_lengths.to(device)

            log_probs = model(images)
            preds = torch.argmax(log_probs, dim=2).permute(1, 0)

            cumulative_lengths = torch.cumsum(label_lengths, dim=0)
            start_indices = cumulative_lengths - label_lengths
            end_indices = cumulative_lengths

            for i in range(preds.size(0)):
                target = labels[start_indices[i].item(): end_indices[i].item()].cpu().numpy()
                pred = decode_ctc_output(preds[i].cpu().numpy())
                random_pred = torch.randint(1, 27, (len(target),))
                total_correct += sum(p == t for p, t in zip(pred, target))
                random_correct += sum(p == t for p, t in zip(random_pred, target))
                total_chars += len(target)

    avg_chars = total_chars / len(test_dataset)
    avg_pred_correct = total_correct / len(test_dataset)
    avg_random_correct = random_correct / len(test_dataset)
    pred_accuracy = total_correct / total_chars
    random_accuracy = random_correct / total_chars
    print(f'Average Number of Total Characters: {avg_chars:.4f}')
    print(f'[Prediction] Average Number of Correct Characters: {avg_pred_correct:.4f}')
    print(f'[Prediction] Accuracy: {pred_accuracy:.4f}')
    print(f'[Baseline] Average Number of Correct Characters: {avg_random_correct:.4f}')
    print(f'[Baseline] Accuracy: {random_accuracy:.4f}')

    print()


def speech_digit_recognition() -> None:
    """ Applies HMM on Free Spoken Digit Dataset to recognize spoken digits from audio signals. """

    # Log function call
    print('--- speech_digit_recognition')

    # Plot MFCC features for recordings of same digit
    output_path = 'figures/mfcc_features_same_digit.png'
    fig, axs = plt.subplots(5, 2, figsize=(14, 20))
    fig.suptitle('MFCC Features for Recordings of Same Digit')

    for idx in range(10):
        y, sr = librosa.load(f'{PROJECT_DIR}/data/external/fsdd/0_george_{idx}.wav')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)
        spec = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=axs[idx // 2][idx % 2])
        axs[idx // 2][idx % 2].set_title(f'Recording {idx}')
        fig.colorbar(spec, ax=axs[idx // 2][idx % 2])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)

    # Plot MFCC features for recordings of different digit
    output_path = 'figures/mfcc_features_diff_digit.png'
    fig, axs = plt.subplots(5, 2, figsize=(14, 20))
    fig.suptitle('MFCC Features for Recordings of Different Digits')

    for idx in range(10):
        y, sr = librosa.load(f'{PROJECT_DIR}/data/external/fsdd/{idx}_george_0.wav')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)
        spec = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=axs[idx // 2][idx % 2])
        axs[idx // 2][idx % 2].set_title(f'Digit {idx}')
        fig.colorbar(spec, ax=axs[idx // 2][idx % 2])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)

    # Extract train features and train model
    list_persons = ['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler']
    list_hmm = []
    for digit in range(10):
        print(f'Training for digit {digit}')
        features = []
        for person in list_persons:
            for recording in range(40):
                y, sr = librosa.load( \
                            f'{PROJECT_DIR}/data/external/fsdd/{digit}_{person}_{recording}.wav')
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15).T
                features.append(mfcc)
        model = hmm.GaussianHMM(n_components=5)
        model.fit(np.vstack(features))
        list_hmm.append(model)

    # Extract test features and evaluate model
    list_actual = []
    list_prediction = []
    for digit in range(10):
        print(f'Evaluating for digit {digit}')
        for person in list_persons:
            for recording in range(40, 50):
                y, sr = librosa.load( \
                            f'{PROJECT_DIR}/data/external/fsdd/{digit}_{person}_{recording}.wav')
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15).T
                list_actual.append(digit)
                list_prediction.append(np.argmax([ model.score(mfcc) for model in list_hmm ]))

    test_accuracy = (np.array(list_actual) == np.array(list_prediction)).mean()
    print(f'Accuracy on test set: {test_accuracy}')

    # Evaluate model on personal recordings
    num_correct = 0
    for digit in range(10):
        y, sr = librosa.load(f'{PROJECT_DIR}/data/interim/5/spoken_digits/{digit}.wav')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15).T
        if np.argmax([ model.score(mfcc) for model in list_hmm ]) == digit:
            print(digit)
            num_correct += 1

    personal_accuracy = num_correct / 10
    print(f'Accuracy on personal set: {personal_accuracy}')

    print()


# pylint: disable=duplicate-code

# pylint: disable-next=too-many-arguments, too-many-positional-arguments
def train_val_test_split(
    X: npt.NDArray, y: npt.NDArray, train_size: float = 0.8, val_size: float = 0.1,
    test_size: float = 0.1, random_seed: int | None = 0
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """ Partitions dataset represented as a pair of array, into three groups. """

    # Reinitialize the random number generator
    if random_seed is not None:
        np.random.seed(random_seed)

    # Ensure the sizes form a probability simplex
    assert train_size + val_size + test_size == 1.0, \
                                    'train_size, val_size, and test_size sizes must sum to 1.'
    assert 0.0 <= train_size <= 1.0, 'train_size must lie in (0, 1)'
    assert 0.0 <= val_size <= 1.0, 'val_size must lie in (0, 1)'
    assert 0.0 <= test_size <= 1.0, 'test_size must lie in (0, 1)'

    # Ensure that X and y are of same length
    assert X.shape[0] == y.shape[0], 'Expected X and y to be the same length'

    # Shuffle the indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Shuffle the dataset as per the indices
    X = X[indices]
    y = y[indices]

    # Compute the splitting indices
    train_end = int(train_size * X.shape[0])
    val_end = train_end + int(val_size * X.shape[0])

    # Split the data
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

# pylint: enable=duplicate-code


if __name__ == '__main__':

    # 2 KDE
    kernel_density_estimation()

    # 3 HMMs
    speech_digit_recognition()

    # 4 RNNs
    bit_counting()
    optimal_character_recognition()
