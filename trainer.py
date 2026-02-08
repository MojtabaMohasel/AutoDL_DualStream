import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import *

import numpy as np
import utils
import os
import json
from model_architectures import ModelBuilder

import tensorflow as tf



def _save_final_report(report, run_dir):
    """Saves the final classification report to a single JSON file."""
    output_dir = os.path.join(run_dir)
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "final_test_results.json")

    new_report = {
        "final_test_report": report,
        "timestamp": utils.get_timestamp()
    }

    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(new_report)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved and appended final classification report to {file_path}")
    except IOError as e:
        print(f"Error saving final classification report to {file_path}: {e}")





def retrain_and_evaluate_best_model(best_individual_parts, best_individual_fitness, config, logger, run_dir, epochs_retrain=50):
    """
    Loads the dataset for the best individual, retrains the model, evaluates it, and saves the results.
    This version correctly decodes the unified chromosome structure.
    """
    logger.info("--- Starting Final Retraining and Evaluation on Best Model ---")


    data_part = best_individual_parts[0]
    win_idx, over_idx = data_part[0], data_part[1]
    win_size_s = config.data_preprocessing.WINDOW_SIZE_SECONDS_OPTIONS[win_idx]
    overlap_r = config.data_preprocessing.OVERLAP_RATIO_OPTIONS[over_idx]
    data_filename = f"data_ws_{win_size_s}s_ol_{overlap_r}.npz"
    data_filepath = os.path.join(config.data_preprocessing.DATA_CACHE_DIR, data_filename)

    logger.info(f"Loading best dataset configuration: {data_filename}")
    try:
        best_data = np.load(data_filepath, allow_pickle=True)
        final_data_dict = {key: best_data[key] for key in best_data}
    except FileNotFoundError:
        logger.error(f"Could not find the dataset file: {data_filepath}. Aborting.")
        return

    num_classes = int(final_data_dict["num_classes"])



    from genetic_optimizer import ChromosomeHelper
    temp_chromosome_helper = ChromosomeHelper(num_classes, config, logger)


    _data_part, hp1d, hp2d, hp_lstm, hp_mlp, batch_idx_part = best_individual_parts

    batch_size = config.ga.BATCH_SIZES_OPTIONS[batch_idx_part[0]]
    loss_gene = data_part[2]
    gamma = config.loss_function.GAMMA_OPTIONS[data_part[3]]
    class_weights = temp_chromosome_helper.get_class_weights_from_chromosome(data_part, config.data_preprocessing.DATA_CACHE_DIR)


    model_builder = ModelBuilder(
        input_shape_1d=final_data_dict["X_time_train"].shape[1:],
        input_shape_2d=final_data_dict["X_freq_train"].shape[1:],
        num_classes=num_classes,
        config=config,
        logger=logger
    )

    best_model = model_builder.build_model(
        hp1d, hp2d, hp_lstm, hp_mlp, loss_gene, gamma, class_weights
    )


    logger.info(f"Combining train and validation sets for final retraining ({epochs_retrain} epochs)...")


    y_train_full = np.concatenate((final_data_dict["y_train"], final_data_dict["y_valid"]), axis=0)
    y_train_full_cat = tf.keras.utils.to_categorical(y_train_full, num_classes=num_classes)


    X_train_full_input = []
    active_streams = config.ga.ACTIVE_STREAMS
    if active_streams.cnn_1d or active_streams.lstm:
        X_time_train_full = np.concatenate((final_data_dict["X_time_train"], final_data_dict["X_time_valid"]), axis=0)
        X_train_full_input.append(X_time_train_full)
    if active_streams.cnn_2d:
        X_freq_train_full = np.concatenate((final_data_dict["X_freq_train"], final_data_dict["X_freq_valid"]), axis=0)
        X_train_full_input.append(X_freq_train_full)


    if len(X_train_full_input) == 1:
        X_train_full_input = X_train_full_input[0]

    fit_kwargs_retrain = {}
    if loss_gene == 0 and class_weights is not None:
        fit_kwargs_retrain['class_weight'] = dict(zip(range(num_classes), class_weights))

    best_model.fit(
        X_train_full_input, y_train_full_cat,
        epochs=epochs_retrain, batch_size=batch_size,
        verbose=1, **fit_kwargs_retrain
    )
    logger.info("Final retraining finished.")


    logger.info("--- Final Evaluation on Test Set ---")
    y_test_true_labels = final_data_dict["y_test"].flatten()

    X_test_input = []
    if active_streams.cnn_1d or active_streams.lstm:
        X_test_input.append(final_data_dict["X_time_test"])
    if active_streams.cnn_2d:
        X_test_input.append(final_data_dict["X_freq_test"])
    if len(X_test_input) == 1:
        X_test_input = X_test_input[0]

    if y_test_true_labels.shape[0] > 0 and (isinstance(X_test_input, np.ndarray) and X_test_input.shape[0] > 0 or isinstance(X_test_input, list) and X_test_input[0].shape[0] > 0):
        y_pred_test_probs = best_model.predict(X_test_input)
        y_pred_test_labels = np.argmax(y_pred_test_probs, axis=1)
        from sklearn.metrics import classification_report
        logger.info("Final Test Classification Report:\n" +
                     classification_report(y_test_true_labels, y_pred_test_labels,
                                           labels=list(range(num_classes)),
                                           target_names=[str(l) for l in final_data_dict["class_labels"]],
                                           digits=4, zero_division=0))
        test_report_dict = classification_report(y_test_true_labels, y_pred_test_labels,
                                                 labels=list(range(num_classes)),
                                                 target_names=[str(l) for l in final_data_dict["class_labels"]],
                                                 digits=4, zero_division=0, output_dict=True)
        _save_final_report(test_report_dict, run_dir)
    else:
        logger.warning("Test data is empty or malformed. Skipping final evaluation.")


    params_to_save = {
        "full_chromosome": best_individual_parts,
        "decoded_win_size_seconds": win_size_s,
        "decoded_overlap_ratio": overlap_r,
        "decoded_batch_size": batch_size,
        "retrained_epochs": epochs_retrain,
        "original_ga_fitness (validation)": best_individual_fitness
    }
    utils.save_model_and_params(best_model, "final_retrained_model", params_to_save, base_dir=run_dir)

    tf.keras.backend.clear_session()



class ModelTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger






    def _calculate_auc_roc_macro(self, y_true, y_pred_probs, num_classes):
        """Calculates macro-averaged AUC ROC for multi-class classification."""
        if num_classes <= 1:
            return 0.0
        if len(np.unique(y_true)) < 2:
            self.logger.warning("Only one class present in y_true. ROC AUC score is not defined in that case. Returning 0.0")
            return 0.0
        try:
            all_possible_labels = list(range(num_classes))


            return roc_auc_score(
                y_true,
                y_pred_probs,
                multi_class='ovr',
                average='macro',
                labels=all_possible_labels
            )
        except ValueError as e:
            self.logger.warning(f"Could not calculate AUC ROC score: {e}. Returning 0.0")
            return 0.0

    def evaluate_on_validation_set(self, model, data_dict):
        """
        Evaluates a model on BOTH validation and test sets.
        Returns: (val_report, val_cm, test_report, test_cm)
        """


        num_classes = int(data_dict["num_classes"])
        all_labels = list(range(num_classes))
        target_names = [str(l) for l in data_dict["class_labels"]]
        active_streams = self.config.ga.ACTIVE_STREAMS






        X_valid_input = []
        if active_streams.cnn_1d or active_streams.lstm:
            X_valid_input.append(data_dict["X_time_valid"])
        if active_streams.cnn_2d:
            X_valid_input.append(data_dict["X_freq_valid"])
        if len(X_valid_input) == 1:
            X_valid_input = X_valid_input[0]


        if (isinstance(X_valid_input, list) and X_valid_input[0].shape[0] == 0) or \
           (isinstance(X_valid_input, np.ndarray) and X_valid_input.shape[0] == 0):
            val_report = {"error": "Empty validation set"}
            val_cm = []
        else:

            y_pred_probs = model.predict(X_valid_input, verbose=0)
            y_pred_labels = np.argmax(y_pred_probs, axis=1)
            y_true_labels = data_dict["y_valid"].flatten()

            val_report = classification_report(y_true_labels, y_pred_labels,
                                            labels=all_labels,
                                            target_names=target_names,
                                            digits=4, zero_division=0, output_dict=True)

            val_cm = confusion_matrix(y_true_labels, y_pred_labels, labels=all_labels).tolist()






        X_test_input = []
        if active_streams.cnn_1d or active_streams.lstm:
            X_test_input.append(data_dict["X_time_test"])
        if active_streams.cnn_2d:
            X_test_input.append(data_dict["X_freq_test"])
        if len(X_test_input) == 1:
            X_test_input = X_test_input[0]


        if (isinstance(X_test_input, list) and X_test_input[0].shape[0] == 0) or \
           (isinstance(X_test_input, np.ndarray) and X_test_input.shape[0] == 0):
            test_report = {"error": "Empty test set"}
            test_cm = []
        else:

            y_pred_test_probs = model.predict(X_test_input, verbose=0)
            y_pred_test_labels = np.argmax(y_pred_test_probs, axis=1)
            y_true_test_labels = data_dict["y_test"].flatten()

            test_report = classification_report(y_true_test_labels, y_pred_test_labels,
                                             labels=all_labels,
                                             target_names=target_names,
                                             digits=4, zero_division=0, output_dict=True)

            test_cm = confusion_matrix(y_true_test_labels, y_pred_test_labels, labels=all_labels).tolist()


        return val_report, val_cm, test_report, test_cm

    def train_and_evaluate(self, model, data_dict, batch_size, epochs,
                           loss_type_gene, class_weights_values, fit_verbose, run_dir):
        """
        Trains the model and evaluates it on validation and test sets.
        This version correctly prepares input data based on ACTIVE_STREAMS.
        """
        y_dev_cat = to_categorical(data_dict["y_dev"], num_classes=data_dict["num_classes"])
        y_valid_cat = to_categorical(data_dict["y_valid"], num_classes=data_dict["num_classes"])


        active_streams = self.config.ga.ACTIVE_STREAMS
        X_dev_input, X_valid_input, X_test_input = [], [], []

        if active_streams.cnn_1d or active_streams.lstm:
            X_dev_input.append(data_dict["X_time_dev"])
            X_valid_input.append(data_dict["X_time_valid"])
            X_test_input.append(data_dict["X_time_test"])
            self.logger.info(f"Using 1D data stream (shape: {data_dict['X_time_dev'].shape})")

        if active_streams.cnn_2d:
            X_dev_input.append(data_dict["X_freq_dev"])
            X_valid_input.append(data_dict["X_freq_valid"])
            X_test_input.append(data_dict["X_freq_test"])
            self.logger.info(f"Using 2D data stream (shape: {data_dict['X_freq_dev'].shape})")


        if len(X_dev_input) == 1:
            X_dev_input = X_dev_input[0]
            X_valid_input = X_valid_input[0]
            X_test_input = X_test_input[0]


        self.logger.info(f"Starting training: epochs={epochs}, batch_size={batch_size}")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
            utils.TimeStopping(seconds=self.config.training.TRAINING_TIMEOUT_SECONDS, verbose=1)
        ]

        fit_kwargs = {}
        if loss_type_gene == 0:
            if class_weights_values is not None and len(class_weights_values) == data_dict["num_classes"]:
                fit_kwargs['class_weight'] = dict(zip(range(data_dict["num_classes"]), class_weights_values))
                self.logger.info(f"Using class_weight in fit: {fit_kwargs['class_weight']}")

        try:
            history = model.fit(
                X_dev_input, y_dev_cat,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                validation_data=(X_valid_input, y_valid_cat),
                verbose=fit_verbose,
                **fit_kwargs
            )
        except Exception:
            self.logger.exception("Training disrupted")
            return 0
        self.logger.info("Training completed.")


        self.logger.info("--- Validation Set Evaluation ---")
        y_pred_valid_probs = model.predict(X_valid_input)
        y_pred_valid_labels = np.argmax(y_pred_valid_probs, axis=1)
        y_true_valid_labels = data_dict["y_valid"].flatten()


        val_report_dict = classification_report(y_true_valid_labels, y_pred_valid_labels,
                                                labels=list(range(len(data_dict["class_labels"]))),
                                                target_names=[str(l) for l in data_dict["class_labels"]],
                                                digits=4, zero_division=0, output_dict=True)

        metric_choice = self.config.ga.OPTIMIZATION_METRIC
        fitness_value = 0.0

        if metric_choice == "accuracy":
            fitness_value = val_report_dict.get("accuracy", 0.0)
        elif metric_choice == "f1_macro":
            fitness_value = val_report_dict.get("macro avg", {}).get("f1-score", 0.0)
        elif metric_choice == "precision_macro":
            fitness_value = val_report_dict.get("macro avg", {}).get("precision", 0.0)
        elif metric_choice == "recall_macro":
            fitness_value = val_report_dict.get("macro avg", {}).get("recall", 0.0)
        elif metric_choice == "auc_roc_macro":
            fitness_value = self._calculate_auc_roc_macro(y_true_valid_labels, y_pred_valid_probs, data_dict["num_classes"])
        else:
            self.logger.warning(f"Unknown OPTIMIZATION_METRIC: '{metric_choice}'. Defaulting to 'f1_macro'.")
            fitness_value = val_report_dict.get("macro avg", {}).get("f1-score", 0.0)




        self.logger.info("Validation Classification Report:\n" +
                         classification_report(y_true_valid_labels, y_pred_valid_labels,
                                               labels = list(range(len(data_dict["class_labels"]))),
                                               target_names=[str(l) for l in data_dict["class_labels"]],
                                               digits=4, zero_division=0))
        self.logger.info("Validation Confusion Matrix:\n" +
                         str(confusion_matrix(y_true_valid_labels, y_pred_valid_labels)))

        self.logger.info("--- Test Set Evaluation ---")
        y_pred_test_probs = model.predict(X_test_input)
        y_pred_test_labels = np.argmax(y_pred_test_probs, axis=1)
        y_true_test_labels = data_dict["y_test"].flatten()

        test_f1_macro = f1_score(y_true_test_labels, y_pred_test_labels, average="macro", zero_division=0)
        self.logger.info(f"Test F1-score (macro): {test_f1_macro:.4f}")
        self.logger.info("Test Classification Report:\n" +
                         classification_report(y_true_test_labels, y_pred_test_labels,
                                               labels = list(range(len(data_dict["class_labels"]))),
                                               target_names=[str(l) for l in data_dict["class_labels"]],
                                               digits=4, zero_division=0))
        self.logger.info("Test Confusion Matrix:\n" +
                         str(confusion_matrix(y_true_test_labels, y_pred_test_labels)))

        val_report_dict = classification_report(y_true_valid_labels, y_pred_valid_labels,
                                               labels = list(range(len(data_dict["class_labels"]))),
                                               target_names=[str(l) for l in data_dict["class_labels"]],
                                               digits=4, zero_division=0, output_dict=True)
        test_report_dict = classification_report(y_true_test_labels, y_pred_test_labels,
                                                  labels = list(range(len(data_dict["class_labels"]))),
                                                  target_names=[str(l) for l in data_dict["class_labels"]],
                                                  digits=4, zero_division=0, output_dict=True)

        self._save_classification_reports(val_report_dict, test_report_dict, run_dir)

        return fitness_value


    def _save_classification_reports(self, val_report, test_report, run_dir):
        """Saves validation and test classification reports to a single JSON file."""
        output_dir = os.path.join(run_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, "test_results.json")


        new_report = {
            "validation_report": val_report,
            "test_report": test_report,
            "timestamp": utils.get_timestamp()
        }


        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = [data]
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []

            data.append(new_report)

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            self.logger.info(f"Successfully saved and appended classification reports to {file_path}")
        except IOError as e:
            self.logger.error(f"Error saving classification reports to {file_path}: {e}")
