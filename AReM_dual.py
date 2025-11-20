import datetime
import glob
import os
import pickle
import random
from functools import lru_cache
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# import tensorflow as tf


from tensorflow.keras.utils import to_categorical
from memoization import cached
from natsort import natsorted
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# from keras_flops import get_flops
# max_flops = 1e7
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
import numpy as np
from scipy.signal import stft
print(tf.__version__)



dataname = 'AReM2'
LOGFILE = 'log-%s-%s.txt' % (dataname, datetime.datetime.now())


fs = 20  # Sampling frequency

N_TRIALS = 2
EPOCHS = 200
PopulationSize = 50
GENERATIONS = 100
max_children = PopulationSize // 4  # Maximum number of children
crossoverProbability = 0.95  # Probability of crossover
mutationProbability = 0.55 # Probability of mutation
FITVERBOSE = 1
batch_sizes = [64, 128, 256, 512, 1024]


MAX_SECONDS = 10
MIN_SECONDS = 0.5
classes = (["standing", "bending2", "cycling", "lying", "sitting", "walking", "bending1"])


path = '/home/v54v562/Imbalanced_GA/%s/' % dataname[:-1]


# To load the object from the file
with open(path + 'all-data-%s.pkl' % dataname[:-1], 'rb') as file:
    ALLDATA = pickle.load(file)


def get_window(X, Y, window_size=20, jump_size=10):
    X_window = []
    y_window = []
    for r in range(0, len(X) - window_size, jump_size):
        x = X[r : r + window_size]
        y = Y[r : r + window_size]
        X_window.append(x)
        y_window.append(stats.mode(y)[0])
    X_window = np.array(X_window)
    y_window = np.array(y_window)

    indices = np.arange(X_window.shape[0])
    # np.random.shuffle(indices)
    return X_window[indices], y_window[indices]

def compute_stft_per_window(X, fs, nperseg):
    """
    Compute the STFT for each window and each feature in the dataset.

    Parameters:
        X (numpy.ndarray): Input data of shape (num_windows, window_size, num_features).
        fs (int): Sampling frequency of the IMU data.
        nperseg (int): Length of each segment for STFT.

    Returns:
        numpy.ndarray: STFT magnitude spectrograms of shape
                       (num_windows, num_features, time_bins, freq_bins).
    """
    num_windows, window_size, num_features = X.shape
    stft_results = []

    for window in range(num_windows):
        window_spectrograms = []
        for feature in range(num_features):
            # Compute STFT for the current feature in the current window
            _, _, Zxx = stft(X[window, :, feature], fs=fs, nperseg=nperseg)
            window_spectrograms.append(np.abs(Zxx))  # Use magnitude |Zxx|

        # Stack spectrograms for all features in this window
        stft_results.append(
            np.stack(window_spectrograms, axis=0)
        )  # Shape: (num_features, time_bins, freq_bins)

    # Combine all windows into a single array
    return np.real(np.array(stft_results).transpose(
        0, 2, 3, 1
    ))  # Shape: (num_windows, time_bins, freq_bins, num_features)


# @keras.saving.register_keras_serializable("ConditionalPoolingLayer1D")
class ConditionalPoolingLayer1D(keras.layers.Layer):
    def __init__(self, pool_size, pool_type, **kwargs):
        super(ConditionalPoolingLayer1D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.pool_type = pool_type
        if pool_type == "max":
            self.pooling_layer = layers.MaxPooling1D(pool_size=pool_size)
        elif pool_type == "average":
            self.pooling_layer = layers.AveragePooling1D(pool_size=pool_size)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        output_length = input_shape[1] // self.pool_size

        if output_length > 0:
            return self.pooling_layer(inputs)
        else:
            return inputs

    def get_config(self):
        base_config = super().get_config()
        config = {
            "pool_size": keras.saving.serialize_keras_object(self.pool_size),
            "pool_type": keras.saving.serialize_keras_object(self.pool_type),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        pool_size_config = config.pop("pool_size")
        pool_type_config = config.pop("pool_type")
        pool_size = keras.saving.deserialize_keras_object(pool_size_config)
        pool_type = keras.saving.deserialize_keras_object(pool_type_config)
        return cls(pool_size, pool_type, **config)

# @keras.saving.register_keras_serializable("ConditionalPoolingLayer2D")
class ConditionalPoolingLayer2D(keras.layers.Layer):
    def __init__(self, pool_size, pool_type, **kwargs):
        super(ConditionalPoolingLayer2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.pool_type = pool_type
        if pool_type == "max":
            self.pooling_layer = layers.MaxPooling2D(pool_size=pool_size)
        elif pool_type == "average":
            self.pooling_layer = layers.AveragePooling2D(pool_size=pool_size)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        output_height = input_shape[1] // self.pool_size[0]
        output_width = input_shape[2] // self.pool_size[1]

        if output_height > 0 and output_width > 0:
            return self.pooling_layer(inputs)
        else:
            return inputs

    def get_config(self):
        base_config = super().get_config()
        config = {
            "pool_size": keras.saving.serialize_keras_object(self.pool_size),
            "pool_type": keras.saving.serialize_keras_object(self.pool_type),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        pool_size_config = config.pop("pool_size")
        pool_type_config = config.pop("pool_type")
        pool_size = keras.saving.deserialize_keras_object(pool_size_config)
        pool_type = keras.saving.deserialize_keras_object(pool_type_config)
        return cls(pool_size, pool_type, **config)


def CombinedIMUModel(
    chromosome1d, chromosome2d, input_shape_1d, input_shape_2d, output_shape, weights = 1., loss = 0, gamma = 2.
):
    """
    Builds a combined model that processes both 1D and 2D inputs, merges features, and applies an MLP.
    """
    # Activation and padding configurations
    Activations = ["", "relu", "tanh", "sigmoid"]
    Padding = ["", "valid", "same"]

    # 1D CNN Branch
    input_1d = layers.Input(shape=input_shape_1d, name="input_1d")
    x1 = layers.BatchNormalization()(input_1d)

    for Block in range(10):
        if chromosome1d[0 + 9 * Block]:
            x1 = layers.Conv1D(
                filters=int(chromosome1d[1 + 9 * Block]),
                kernel_size=int(chromosome1d[2 + 9 * Block]),
                activation=Activations[int(chromosome1d[3 + 9 * Block])],
                padding=Padding[int(chromosome1d[4 + 9 * Block])],
                strides=int(chromosome1d[5 + 9 * Block]),
            )(x1)
            if chromosome1d[6 + 9 * Block]:
                x1 = layers.BatchNormalization()(x1)
            if chromosome1d[7 + 9 * Block]:
                pool_type = "max" if chromosome1d[8 + 9 * Block] else "average"
                x1 = ConditionalPoolingLayer1D(2, pool_type=pool_type)(x1)
                # x1 = layers.MaxPooling1D(pool_size=2)(x1) if pool_type == "max" else layers.AveragePooling1D(pool_size=2)(x1)

    if chromosome1d[90]:
        x1 = layers.GlobalAveragePooling1D()(x1)
    else:
        x1 = layers.Flatten()(x1)

    if chromosome1d[91]:
        x1 = layers.Dropout(chromosome1d[92])(x1)

    # 2D CNN Branch
    input_2d = layers.Input(shape=input_shape_2d, name="input_2d")
    x2 = layers.BatchNormalization()(input_2d)

    for Block in range(10):
        if chromosome2d[0 + 9 * Block]:
            x2 = layers.Conv2D(
                filters=int(chromosome2d[1 + 9 * Block]),
                kernel_size=(
                    int(chromosome2d[2 + 9 * Block]),
                    int(chromosome2d[2 + 9 * Block]),
                ),
                activation=Activations[int(chromosome2d[3 + 9 * Block])],
                padding=Padding[int(chromosome2d[4 + 9 * Block])],
                strides=(
                    int(chromosome2d[5 + 9 * Block]),
                    int(chromosome2d[5 + 9 * Block]),
                ),
            )(x2)
            if chromosome2d[6 + 9 * Block]:
                x2 = layers.BatchNormalization()(x2)
            if chromosome2d[7 + 9 * Block]:
                pool_type = "max" if chromosome2d[8 + 9 * Block] else "average"
                x2 = ConditionalPoolingLayer2D((2, 2), pool_type=pool_type)(x2)
                # x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2) if pool_type == "max" else layers.AveragePooling2D(pool_size=(2, 2))(x2)

    if chromosome2d[90]:
        x2 = layers.GlobalAveragePooling2D()(x2)
    else:
        x2 = layers.Flatten()(x2)

    if chromosome2d[91]:
        x2 = layers.Dropout(chromosome2d[92])(x2)

    # Feature Concatenation
    merged_features = layers.Concatenate(name="concatenated_features")([x1, x2])

    # Shared MLP
    x = merged_features
    for Block in range(3):
        if chromosome1d[93 + Block * 7]:  # Shared MLP uses chromosome1d for design
            if chromosome1d[94 + Block * 7]:
                x = layers.BatchNormalization()(x)
            if chromosome1d[95 + Block * 7]:
                x = layers.Dense(
                    units=chromosome1d[96 + Block * 7],
                    activation=Activations[int(chromosome1d[97 + Block * 7])],
                )(x)
            if chromosome1d[98 + Block * 7]:
                x = layers.Dropout(chromosome1d[99 + Block * 7])(x)

    # Output Layer
    output = layers.Dense(
        units=output_shape,
        activation="softmax",
        kernel_regularizer=keras.regularizers.l1(chromosome1d[114]),
    )(x)

    # Compile Model
    model = models.Model(
        inputs=[input_1d, input_2d], outputs=output, name="CombinedIMUModel"
    )
    learning_rate = chromosome1d[115]
    if loss == 0:
        lss = keras.losses.CategoricalCrossentropy()
    elif loss == 1:
        lss = keras.losses.CategoricalFocalCrossentropy(gamma=gamma, alpha=weights)
    else:
        raise ValueError(str(loss))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=lss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc", curve="PR"),
        ],
    )

    return model


def train(
    cnn,
    X_time_train,
    X_freq_train,
    y_train,
    X_time_test,
    X_freq_test,
    y_test,
    X_time_valid,
    X_freq_valid,
    y_valid,
    BATCH_SIZE,
    weights,
    loss
):
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    y_test = to_categorical(y_test)
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
    if loss == 0:
        kw = {'class_weight':dict(zip(range(len(weights)), weights))}
    else:
        kw = {}
    history = cnn.fit(
        [X_time_train, X_freq_train],
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[callback],
        validation_data=([X_time_valid, X_freq_valid], y_valid),
        verbose=FITVERBOSE,
        **kw
    )

    y_pred = cnn([X_time_valid, X_freq_valid]).numpy().argmax(axis=1)
    y_pred2 = cnn([X_time_test, X_freq_test]).numpy().argmax(axis=1)
    print(
        "Valid",
        f1_score(y_valid.argmax(axis=1), y_pred, average="macro"),
        file=open(LOGFILE, "a"),
    )
    print(
        classification_report(y_valid.argmax(axis=1), y_pred, digits=4),
        file=open(LOGFILE, "a"),
    )
    print(confusion_matrix(y_valid.argmax(axis=1), y_pred), file=open(LOGFILE, "a"))
    print(
        (
            confusion_matrix(y_valid.argmax(axis=1), y_pred, normalize="true") * 100
        ).round(2),
        file=open(LOGFILE, "a"),
    )

    print(
        "Test",
        f1_score(y_test.argmax(axis=1), y_pred2, average="macro"),
        file=open(LOGFILE, "a"),
    )
    print(
        classification_report(y_test.argmax(axis=1), y_pred2, digits=4),
        file=open(LOGFILE, "a"),
    )
    print(confusion_matrix(y_test.argmax(axis=1), y_pred2), file=open(LOGFILE, "a"))
    print(
        (
            confusion_matrix(y_test.argmax(axis=1), y_pred2, normalize="true") * 100
        ).round(2),
        file=open(LOGFILE, "a"),
    )

    return f1_score(y_valid.argmax(axis=1), y_pred, average="macro")


# parent selection function
# tournament technique
def parent_selection():
    # Initialize the best candidate as None
    best_candidate = None

    # Sample a subset of the population
    for _ in range(PopulationSize // 5):
        ch = random.choice(Population)
        # Update best_candidate if it is None or the current chromosome has a lower cost
        if best_candidate is None or ch[-1] > best_candidate[-1]:
            best_candidate = ch

    # Return the chromosome portion without the cost if needed
    return best_candidate[:-1]


def generate_data_chromosome():
    # win + jmp + + loss + gamma + clswghts
    winsize = random.randint(1, int(MAX_SECONDS/ MIN_SECONDS))
    jmp = random.randint(0, 2)
    lss = random.randint(0, 1)
    gmma = random.randint(0, 9)
    out = [winsize,jmp,lss,gmma]
    for _ in range(len(classes)):
        out.append(random.randint(0, 9))
    return out

# class-preserving crossover
def crossover_data(parent1, parent2):
    crossover_point = random.randint(
        1, len(parent1) - 1
    )  # Ensure at least one gene is swapped
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return [child1, child2]


# mutation function
# single swap technique
def mutation_data(chromosome):
    # Randomly select an index to mutate
    index_to_mutate = random.randint(0, len(chromosome) - 1)

    # Mutate the selected gene based on its position
    if index_to_mutate == 0:  # winsize
        chromosome[index_to_mutate] = random.randint(1, int(MAX_SECONDS / MIN_SECONDS))
    elif index_to_mutate == 1:  # jmp
        chromosome[index_to_mutate] = random.randint(0, 2)
    elif index_to_mutate == 2:  # lss
        chromosome[index_to_mutate] = random.randint(0, 1)
    elif index_to_mutate == 3:  # gmma
        chromosome[index_to_mutate] = random.randint(0, 9)
    else:  # clswghts (for each class)
        chromosome[index_to_mutate] = random.randint(0, 9)

    return chromosome


def generate_hp_chromosome():
    # len(vartype), len(varbound)
    individual = []
    for i in range(len(vartype)):
        if vartype[i] == "int":
            # Generate a random integer within the specified bounds
            value = random.randint(varbound[i][0], varbound[i][1])
        elif vartype[i] == "real":
            # Generate a random float within the specified bounds
            value = random.uniform(varbound[i][0], varbound[i][1])
        individual.append(value)
    return individual


def crossover_hp(parent1, parent2):
    """Perform one-point crossover between two parents."""
    crossover_point = random.randint(
        1, len(parent1) - 1
    )  # Ensure at least one gene is swapped
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return [child1, child2]


def mutation_hp(individual):
    """Mutate a single randomly selected gene in an individual with a given mutation rate, respecting variable bounds."""
    # Randomly decide if mutation occurs
    index_to_mutate = random.randint(0, len(individual) - 1)

    if vartype[index_to_mutate] == "int":
        # Randomly change the integer within its bounds
        individual[index_to_mutate] = random.randint(
            varbound[index_to_mutate][0], varbound[index_to_mutate][1]
        )
    elif vartype[index_to_mutate] == "real":
        # Randomly change the real number within a small range
        mutation_amount = random.uniform(-0.1, 0.1)  # Adjust this range as needed
        new_value = individual[index_to_mutate] + mutation_amount

        # Ensure the new value is within bounds
        new_value = max(
            varbound[index_to_mutate][0], min(new_value, varbound[index_to_mutate][1])
        )
        individual[index_to_mutate] = new_value
    return individual


def generate_hp_chromosome2():
    # len(vartype2), len(varbound2)
    individual = []
    for i in range(len(vartype2)):
        if vartype2[i] == "int":
            # Generate a random integer within the specified bounds
            value = random.randint(varbound2[i][0], varbound2[i][1])
        elif vartype2[i] == "real":
            # Generate a random float within the specified bounds
            value = random.uniform(varbound2[i][0], varbound2[i][1])
        individual.append(value)
    return individual


def crossover_hp2(parent1, parent2):
    """Perform one-point crossover between two parents."""
    crossover_point = random.randint(
        1, len(parent1) - 1
    )  # Ensure at least one gene is swapped
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return [child1, child2]


def mutation_hp2(individual):
    """Mutate a single randomly selected gene in an individual with a given mutation rate, respecting variable bounds."""
    # Randomly decide if mutation occurs
    index_to_mutate = random.randint(0, len(individual) - 1)

    if vartype2[index_to_mutate] == "int":
        # Randomly change the integer within its bounds
        individual[index_to_mutate] = random.randint(
            varbound2[index_to_mutate][0], varbound2[index_to_mutate][1]
        )
    elif vartype2[index_to_mutate] == "real":
        # Randomly change the real number within a small range
        mutation_amount = random.uniform(-0.1, 0.1)  # Adjust this range as needed
        new_value = individual[index_to_mutate] + mutation_amount

        # Ensure the new value is within bounds
        new_value = max(
            varbound2[index_to_mutate][0], min(new_value, varbound2[index_to_mutate][1])
        )
        individual[index_to_mutate] = new_value
    return individual


def mutation(ch):
    assert (
        len(ch) == len_data + len_hp + len_hp2 + 1 + 1
    ), "len is %d but must be %d" % (
        len(ch),
        len_data + len_hp + len_hp2 + 1 + 1,
    )
    ch[:len_data] = mutation_data(ch[:len_data])
    ch[len_data : len_data + len_hp] = mutation_hp(ch[len_data : len_data + len_hp])
    ch[len_data + len_hp : len_data + len_hp + len_hp2] = mutation_hp(
        ch[len_data + len_hp : len_data + len_hp + len_hp2]
    )

    ch[-1] = fitness_function(ch[:-1])


def crossover(p1, p2):
    assert len(p1) == len_data + len_hp + len_hp2 + 1, "len is %d but must be %d" % (
        len(p2),
        len_data + len_hp + len_hp2 + 1,
    )

    assert len(p2) == len_data + len_hp + len_hp2 + 1, "len is %d but must be %d" % (
        len(p2),
        len_data + len_hp + len_hp2 + 1,
    )
    # Perform crossover on the data part of the chromosomes
    c1_data, c2_data = crossover_data(p1[:len_data], p2[:len_data])

    # Perform crossover on the hyperparameter part of the chromosomes
    c1_hp, c2_hp = crossover_hp(
        p1[len_data : len_data + len_hp], p2[len_data : len_data + len_hp]
    )

    c1_hp2, c2_hp2 = crossover_hp2(
        p1[len_data + len_hp : len_data + len_hp + len_hp2],
        p2[len_data + len_hp : len_data + len_hp + len_hp2],
    )

    # Combine both parts to form the new chromosomes
    c1 = c1_data + c1_hp + c1_hp2 + [p2[-1]]
    c2 = c2_data + c2_hp + c2_hp2 + [p1[-1]]

    # Compute fitness for the whole chromosome
    c1_fitness = fitness_function(c1)
    c2_fitness = fitness_function(c2)

    # Append fitness scores to the chromosomes
    c1.append(c1_fitness)
    c2.append(c2_fitness)

    return [c1, c2]

vartype = (
    ["int", "int", "int", "int", "int", "int", "int", "int", "int"] * 10
    + ["int", "int", "real"]
    + ["int", "int", "int", "int", "int", "int", "real"] * 3
    + ["real", "real"]
)
varbound = (
    [[0, 1], [1, 500], [1, 8], [1, 3], [1, 2], [1, 2], [0, 1], [0, 1], [0, 1]] * 10
    + [[0, 1], [0, 1], [0.4, 0.5]]
    + [[0, 1], [0, 1], [0, 1], [3, 1024], [1, 3], [0, 1], [0.4, 0.5]] * 3
    + [[1e-5, 1e-3], [1e-6, 1e-2]]
)
vartype2 = ["int", "int", "int", "int", "int", "int", "int", "int", "int"] * 10 + [
    "int",
    "int",
    "real",
]
varbound2 = [
    [0, 1],
    [1, 500],
    [1, 8],
    [1, 3],
    [1, 2],
    [1, 2],
    [0, 1],
    [0, 1],
    [0, 1],
] * 10 + [[0, 1], [0, 1], [0.4, 0.5]]


# load data from npz file
'arem2-w-20.npz'

# X_time_train, X_time_test, X_time_valid, X_freq_train, X_freq_test, X_freq_valid, y_train, y_test, y_valid

# print(classes)
# print(X_time_train.shape, X_time_test.shape, X_time_valid.shape, y_train.shape, y_test.shape, y_valid.shape)

# labels = y_train.flatten()
# labels_unique = np.unique(labels)
# class_weights = compute_class_weight("balanced", classes=labels_unique, y=labels)
# class_weights = dict(zip(labels_unique, class_weights))
# print(class_weights)
#
# labels = y_valid.flatten()
# labels_unique = np.unique(labels)
# class_weights = compute_class_weight("balanced", classes=labels_unique, y=labels)
# class_weights = dict(zip(labels_unique, class_weights))
# print(class_weights)
#
# labels = y_test.flatten()
# labels_unique = np.unique(labels)
# class_weights = compute_class_weight("balanced", classes=labels_unique, y=labels)
# class_weights = dict(zip(labels_unique, class_weights))
# print(class_weights)

# win + jmp + + loss + gamma + clswghts
len_data = len(generate_data_chromosome())
# 1 + 1 + 1 + 1 + len(np.unique(y_train))
len_hp = len(generate_hp_chromosome())
len_hp2 = len(generate_hp_chromosome2())
# len_hp
# len_data + len_hp

# Apply STFT to each window and feature

# print(X_time_train.shape, X_time_test.shape, X_time_valid.shape, X_freq_train.shape, X_freq_test.shape, X_freq_valid.shape, y_train.shape, y_test.shape, y_valid.shape)



@cached(max_size=300)
def fitness_function(chromosome):
    assert (
        len(chromosome) == len_data + len_hp + len_hp2 + 1
    ), "len is %d but must be %d" % (
        len(chromosome),
        len_data + len_hp + len_hp2 + 1,
    )

    BATCH_SIZE = chromosome[-1]
    winsize, jmp, lss, gmma, *clswghts = chromosome[:len_data]
    WINDOW_SIZE = int(winsize * MIN_SECONDS * fs)
    JUMP_SIZE = int(WINDOW_SIZE * [0.25, 0.5, 0.75][jmp])
    GAMMA = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0][gmma]
    weights = np.array([ [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0][i] for i in clswghts])
    X_time_train, X_time_test, X_time_valid, X_freq_train, X_freq_test, X_freq_valid, y_train, y_test, y_valid = ALLDATA[(winsize, jmp)]
    try:
        model = CombinedIMUModel(
            chromosome[len_data : len_data + len_hp],
            chromosome[len_data + len_hp : len_data + len_hp + len_hp2],
            input_shape_1d=X_time_train[0].shape,
            input_shape_2d=X_freq_train[0].shape,
            output_shape=len(classes),
            weights = weights,
            loss = lss,
            gamma = GAMMA

        )
        print("Bad", chromosome, file=open(LOGFILE, "a"))
    except:
        return 0

    # load X_time_train,...
    if len(X_time_train) < len(np.unique(y_test)):
        return 0

    print("*****" * 10, file=open(LOGFILE, "a"))
    print(datetime.datetime.now(), file=open(LOGFILE, "a"))
    print(chromosome, file=open(LOGFILE, "a"))
    t = 0
    for _ in range(N_TRIALS):
        print("%%%" * 10, file=open(LOGFILE, "a"))
        t += train(
            model,
            X_time_train,
            X_freq_train,
            y_train,
            X_time_test,
            X_freq_test,
            y_test,
            X_time_valid,
            X_freq_valid,
            y_valid,
            BATCH_SIZE=BATCH_SIZE,
            loss=lss,
            weights=weights
        )
        print("^^^" * 10, file=open(LOGFILE, "a"))
    print("#####" * 10, file=open(LOGFILE, "a"))
    return t / N_TRIALS


#
# #
# # # initialization
# Population = []
# while len(Population) < PopulationSize:
#     chromosome = generate_data_chromosome()
#     Population.append(chromosome)
# for i in range(len(Population)):
#     added = False
#     while not added:
#         try:
#             winsize, jmp, lss, gmma, *clswghts = Population[i]
#             X_time_train, X_time_test, X_time_valid, X_freq_train, X_freq_test, X_freq_valid, y_train, y_test, y_valid = ALLDATA[(winsize, jmp)]
#             hpc1 = generate_hp_chromosome()
#             hpc2 = generate_hp_chromosome2()
#             CombinedIMUModel(
#                 hpc1,
#                 hpc2,
#                 input_shape_1d=X_time_train[0].shape,
#                 input_shape_2d=X_freq_train[0].shape,
#                 output_shape=len(classes),
#                 weights = 1,
#                 loss = 0
#             )
#             Population[i].extend(hpc1 + hpc2)
#             added = True
#         except ValueError:
#             pass
#
#     Population[i].append(random.randint(0, len(batch_sizes)))  # BATCH_SIZE
#
# assert len(Population) == PopulationSize, "len is %d but must be %d" % (len(Population), PopulationSize)
#
# for i in range(len(Population)):
#     Population[i].append(fitness_function(Population[i]))


filename = "chkpnts/population-AReM2-9-100-2024-11-30 02:12:11.695353.pkl"
# Check if the file exists
if os.path.exists(filename):
    # Load the Population variable from the file using pickle
    with open(filename, "rb") as file:
        Population = pickle.load(file)
    print(f"Loaded population from {filename}")
else:
    print(f"File {filename} does not exist.")


cost_list = list()
Population.sort(key=lambda q: q[-1], reverse=True)
cost_list.append(list(map(lambda p: p[-1], Population)))
iteration_count = 9
best_cost = Population[0][-1]
patience = 20  # Number of generations to wait for improvement
patience_counter = 0  # Counter for generations without improvement

os.makedirs("chkpnts", exist_ok=True)







print("%dth iteration, current fitness: %.2f" % (iteration_count, best_cost * 100))

for generation in range(iteration_count, GENERATIONS):
    np.array(Population).shape
    filename = f"chkpnts/population-{dataname}-{generation}-{GENERATIONS}-{datetime.datetime.now()}.pkl"

    # Save the Population variable to a file using pickle
    with open(filename, "wb") as file:
        pickle.dump(Population, file)

    print("Population, Generation", generation, file=open(LOGFILE, "a"))
    print(Population, file=open(LOGFILE, "a"))
    random.shuffle(Population)
    # recombine parents
    new_children = list()
    for _ in range(max_children):
        p1, p2 = parent_selection(), parent_selection()
        while p1 == p2:
            p2 = parent_selection()
        done = False
        if random.random() < crossoverProbability:
            children = crossover(p1, p2)
            done = True
        else:
            children = [p1[:] + [0], p2[:] + [0]]
        for child in children:
            if random.random() < mutationProbability or not done:
                mutation(child)
            new_children.append(child)
    Population.extend(new_children)

    # kill people with upper cost (goal: minimizing cost)
    Population.sort(key=lambda q: q[-1], reverse=True)
    del Population[PopulationSize:]

    cost_list.append(list(map(lambda p: p[-1], Population)))
    iteration_count += 1
    current_best_cost = Population[0][-1]

    # Check for improvement
    if current_best_cost < best_cost:
        best_cost = current_best_cost
        patience_counter = 0  # Reset counter if there's an improvement
    else:
        patience_counter += 1  # Increment counter if no improvement

    if patience_counter >= patience:
        print(
            "Early stopping at generation %d, no improvement in last %d generations."
            % (generation, patience)
        )
        break

    if iteration_count % 1 == 0:
        print(
            "%03dth iteration, current fitness: %.2f"
            % (iteration_count, 100 * current_best_cost)
        )

    print("cost", cost_list, file=open(LOGFILE, "a"))

print("FINAL", file=open(LOGFILE, "a"))

print(Population, file=open(LOGFILE, "a"))
print("cost", cost_list, file=open(LOGFILE, "a"))

print('Best Valid', cost_list[0], file=open(LOGFILE, "a"))
print(Population[0], file=open(LOGFILE, "a"))

print("%dth iteration, current cost: %.2f" % (iteration_count, 100 * Population[0][-1]))
iteration = range(len(cost_list))
plt.plot(
    iteration,
    cost_list,
    label=["Best"] + ["" for i in range(1, PopulationSize - 1)] + ["Worst"],
)
plt.xlabel("Generation")
plt.ylabel("macro F1 score")
plt.legend()
plt.savefig('%s-1.pdf' % dataname, bbox_inches='tight')

# Calculate the mean along the second dimension (dim 12)
mean = np.mean(cost_list, axis=1)

# Calculate the standard error of the mean
sem = np.std(cost_list, axis=1) / np.sqrt(np.array(cost_list).shape[1])

# Calculate the 95% confidence interval
ci = 1.96 * sem  # 1.96 is the z-score for 95% CI

# Create the upper and lower bounds of the CI
upper_bound = mean + ci
lower_bound = mean - ci

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(mean, label="Mean", marker="o")
plt.fill_between(
    range(len(mean)), lower_bound, upper_bound, color="b", alpha=0.2, label="95% CI"
)
plt.title("Mean ± 95% Confidence Interval")
plt.ylabel("Mean Value")
plt.xticks(range(len(mean)))
plt.legend()
plt.grid()
plt.savefig('%s-2.pdf' % dataname, bbox_inches='tight')
