import logging
import os
import datetime
import numpy as np
import tensorflow as tf

import datetime
import time
from typeguard import typechecked
import json
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


@tf.keras.utils.register_keras_serializable(package="xAutoGA")
class TimeStopping(Callback):
    """Stop training when a specified amount of time has passed.

    Args:
        seconds: maximum amount of time before stopping.
            Defaults to 86400 (1 day).
        verbose: verbosity mode. Defaults to 0.
    """

    @typechecked
    def __init__(self, seconds: int = 86400, verbose: int = 0):
        super().__init__()

        self.seconds = seconds
        self.verbose = verbose
        self.stopped_epoch = None

    def on_train_begin(self, logs=None):
        self.stopping_time = time.time() + self.seconds

    def on_epoch_end(self, epoch, logs={}):
        if time.time() >= self.stopping_time:
            self.model.stop_training = True
            self.stopped_epoch = epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch is not None and self.verbose > 0:
            formatted_time = datetime.timedelta(seconds=self.seconds)
            msg = "Timed stopping at epoch {} after training for {}".format(
                self.stopped_epoch + 1, formatted_time
            )
            print(msg)

    def get_config(self):
        config = {
            "seconds": self.seconds,
            "verbose": self.verbose,
        }

        base_config = super().get_config()
        return {**base_config, **config}

def get_timestamp():
    """Returns a timestamp string."""
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def setup_logger(run_dir, prefix, level=logging.INFO):
    """Sets up a logger that appends to a file and writes to console."""

    os.makedirs(run_dir, exist_ok=True)


    log_filename = f"{prefix}.log"
    log_filepath = os.path.join(run_dir, log_filename)

    logger = logging.getLogger(prefix)
    logger.setLevel(level)


    fh = logging.FileHandler(log_filepath)
    fh.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    gpu_devices = tf.config.list_physical_devices('GPU')
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Available GPUs: {gpu_devices}")
    if not gpu_devices:
        logger.warning("No GPU detected by TensorFlow. Running on CPU will be slow.")

    return logger

def calculate_mode(y_values):
    """Calculates the mode of a list or array of values."""
    if not isinstance(y_values, np.ndarray):
        y_values = np.array(y_values)
    if y_values.size == 0:
        raise ValueError('y_values should at least has one value but its size is zero.')


    values, counts = np.unique(y_values, return_counts=True)
    return values[np.argmax(counts)]


def save_model_and_params(model, model_name, params_dict, base_dir="best_models"):
    """Saves the Keras model and its parameters."""
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    model_filename = f"{model_name}_{timestamp}.keras"
    model_filepath = os.path.join(base_dir, model_filename)
    model.save(model_filepath)
    print(f"Saved Keras model to {model_filepath}")


    params_filename = f"{model_name}_params_{timestamp}.json"
    params_filepath = os.path.join(base_dir, params_filename)

    def convert_numpy_to_list(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, (np.int32, np.int64, np.float32, np.float64)):
            return item.item()
        if isinstance(item, dict):
            return {k: convert_numpy_to_list(v) for k, v in item.items()}
        if isinstance(item, list):
            return [convert_numpy_to_list(i) for i in item]
        return item

    json_safe_params = convert_numpy_to_list(params_dict)
    with open(params_filepath, 'w') as f:
        json.dump(json_safe_params, f, indent=4)
    print(f"Saved model parameters to {params_filepath}")
