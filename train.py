from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Add, ReLU, LSTM, ConvLSTM2D
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, concatenate, Input, AveragePooling2D, TimeDistributed, \
    Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback

import os
import numpy as np

from config import *

# loss to minimise
def custom_loss(y_true, y_pred):
    # wasd keys
    loss1a = losses.binary_crossentropy(y_true[:, :, 0:4],
                                        y_pred[:, :, 0:4])
    # space key
    loss1b = losses.binary_crossentropy(y_true[:, :, 4:5],
                                        y_pred[:, :, 4:5])

    # left click
    loss2a = losses.binary_crossentropy(y_true[:, :, n_keys:n_keys + 1],
                                        y_pred[:, :, n_keys:n_keys + 1])
    # right click
    loss2b = losses.binary_crossentropy(y_true[:, :, n_keys + 1:n_keys + n_clicks],
                                        y_pred[:, :, n_keys + 1:n_keys + n_clicks])
    # mouse move x
    loss3 = losses.categorical_crossentropy(y_true[:, :, n_keys + n_clicks:n_keys + n_clicks + n_mouse_x],
                                            y_pred[:, :, n_keys + n_clicks:n_keys + n_clicks + n_mouse_x])
    # mouse move y
    loss4 = losses.categorical_crossentropy(
        y_true[:, :, n_keys + n_clicks + n_mouse_x:n_keys + n_clicks + n_mouse_x + n_mouse_y],
        y_pred[:, :, n_keys + n_clicks + n_mouse_x:n_keys + n_clicks + n_mouse_x + n_mouse_y])

    return K.mean(loss1a + loss1b + loss2a + loss2b + loss3 + loss4)


def wasd_acc(y_true, y_pred):
    return keras.metrics.binary_accuracy(y_true[:, :, 0:4], y_pred[:, :, 0:4])

def space_acc(y_true, y_pred):  # space
    return keras.metrics.binary_accuracy(y_true[:, :, 4:5], y_pred[:, :, 4:5])

def Lclk_acc(y_true, y_pred):
    return keras.metrics.binary_accuracy(y_true[:, :, n_keys:n_keys + 1], y_pred[:, :, n_keys:n_keys + 1],
                                         threshold=0.5)

def Rclk_acc(y_true, y_pred):
    return keras.metrics.binary_accuracy(y_true[:, :, n_keys + 1:n_keys + n_clicks],
                                         y_pred[:, :, n_keys + 1:n_keys + n_clicks], threshold=0.5)

def m_x_acc(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true[:, :, n_keys + n_clicks:n_keys + n_clicks + n_mouse_x],
                                              y_pred[:, :, n_keys + n_clicks:n_keys + n_clicks + n_mouse_x])

def m_y_acc(y_true, y_pred):
    return keras.metrics.categorical_accuracy(
        y_true[:, :, n_keys + n_clicks + n_mouse_x:n_keys + n_clicks + n_mouse_x + n_mouse_y],
        y_pred[:, :, n_keys + n_clicks + n_mouse_x:n_keys + n_clicks + n_mouse_x + n_mouse_y])

def build_model(load_weights=True):
    # useful tutorial for building, https://keras.io/getting-started/functional-api-guide/
    print('-- building model from scratch --')

    base_model = EfficientNetB0(weights='imagenet', input_shape=(input_shape[1:]), include_top=False)
    base_model.trainable = True

    # for (i, layers) in enumerate(base_model.layers):
    #    print(i)
    #    print(layers.output.shape)

    intermediate_model = Model(inputs=base_model.input, outputs=base_model.layers[52].output)
    intermediate_model.trainable = True

    input_1 = Input(shape=input_shape, name='main_in')
    x = TimeDistributed(intermediate_model)(input_1)
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), stateful=False, return_sequences=True)(x)
    x = TimeDistributed(Flatten())(x)

    # 3) add shared fc layers
    dense_5 = x

    # 4) set up outputs, sepearate outputs will allow seperate losses to be applied
    output_1 = TimeDistributed(Dense(n_keys, activation='sigmoid'))(dense_5)
    output_2 = TimeDistributed(Dense(n_clicks, activation='sigmoid'))(dense_5)
    output_3 = TimeDistributed(Dense(n_mouse_x, activation='softmax'))(dense_5)
    output_4 = TimeDistributed(Dense(n_mouse_y, activation='softmax'))(dense_5)
    output_5 = TimeDistributed(Dense(1, activation='linear'))(dense_5)

    output_all = concatenate([output_1, output_2, output_3, output_4, output_5], axis=-1)
    model = Model(input_1, output_all)

    # Load weights if they exist
    if load_weights and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        model.load_weights(checkpoint_path)

    print(model.summary())

    opt = optimizers.Adam(learning_rate=l_rate)
    model.compile(loss=custom_loss, optimizer=opt, metrics=[Lclk_acc, Rclk_acc, wasd_acc, space_acc, m_x_acc, m_y_acc])
    print('successfully compiled model')
    return model


# New DataGenerator: pre-indexes all samples across all files and loads only the samples needed per batch using mmap
class DataGenerator(Sequence):
    def __init__(self, input_dir, output_dir, batch_size=32, shuffle=True):
        self.X_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith("_X.npy")])
        self.Y_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_Y.npy")])

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.sample_index = []  # list of (file_idx, sample_idx)
        self.X_mmaps = []
        self.Y_mmaps = []

        for file_idx, (x_path, y_path) in enumerate(zip(self.X_files, self.Y_files)):
            x_data = np.load(x_path, mmap_mode='r')
            y_data = np.load(y_path, mmap_mode='r')
            self.X_mmaps.append(x_data)
            self.Y_mmaps.append(y_data)

            for sample_idx in range(x_data.shape[0]):
                self.sample_index.append((file_idx, sample_idx))

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.sample_index) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.sample_index[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = []
        Y_batch = []

        for file_idx, sample_idx in batch_indices:
            x_sample = self.X_mmaps[file_idx][sample_idx].astype(np.float32) / 255.0
            y_sample = self.Y_mmaps[file_idx][sample_idx].astype(np.float32)
            X_batch.append(x_sample)
            Y_batch.append(y_sample)

        return np.array(X_batch), np.array(Y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)

def generate_data(batch_size=1, shuffle=True):
    input_dir = preprocessed_dataset_dir + "/input"
    output_dir = preprocessed_dataset_dir + "/output"
    return DataGenerator(input_dir, output_dir, batch_size=batch_size, shuffle=shuffle)

def generate_validation_data(batch_size=1, shuffle=True):
    input_dir = validation_dataset_dir + "/input"
    output_dir = validation_dataset_dir + "/output"
    return DataGenerator(input_dir, output_dir, batch_size=batch_size, shuffle=shuffle)


class EpochCheckpoint(Callback):
    def __init__(self, output_dir, save_freq=1):
        super(EpochCheckpoint, self).__init__()
        self.output_dir = output_dir
        self.save_freq = save_freq
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)
            model_path = os.path.join(epoch_dir, "model_160x90_with_dropout.keras")
            self.model.save(model_path)
            print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    # Create saved_model directory if it doesn't exist
    os.makedirs("saved_model", exist_ok=True)

    # Build model and load weights if available
    model = build_model(load_weights=True)
    train_generator = generate_data(batch_size=batch_size)
    print(f"Using data generator with {len(train_generator)} batches")

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=True,
            mode='auto',
            save_freq='epoch'
        ),
        EpochCheckpoint(
            output_dir=checkpoint_dir,
            save_freq=1  # Save every epoch
        )
    ]

    try:
        model.fit(
            train_generator,
            epochs=epochs,
            shuffle=False,  # generator handles shuffling
            validation_data=generate_validation_data(),
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model weights...")
        model.save_weights(checkpoint_path)
        print(f"Weights saved to {checkpoint_path}")
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Attempting to save model weights...")
        model.save_weights(checkpoint_path)
        print(f"Weights saved to {checkpoint_path}")
        raise e

    # Save the final model
    model.save(saved_model_path)
    print("Model saved to " + saved_model_path)