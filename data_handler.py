import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import calculate_mode
import os
from feature_extractor import FeatureExtractor
from sklearn.preprocessing import LabelEncoder


class DataHandler:
    def __init__(self, csv_filepath, config, logger):
        self.csv_filepath = csv_filepath
        self.config = config
        self.logger = logger
        self.raw_df = None
        self.processed_df = None
        self.num_classes = 0
        self.class_labels = None

    def load_data(self):
        self.logger.info(f"Loading data from {self.csv_filepath}")
        try:
            self.raw_df = pd.read_csv(self.csv_filepath)
            self.raw_df = self.raw_df.fillna(0)

            if "LABEL" in self.raw_df.columns and "Class" not in self.raw_df.columns:
                self.raw_df["Class"] = self.raw_df["LABEL"]
                self.raw_df = self.raw_df.drop("LABEL", axis=1)
            if "Subject" not in self.raw_df.columns:

                if 'Participant' in self.raw_df.columns:
                    self.raw_df['Subject'] = self.raw_df['Participant']
                elif 'ID' in self.raw_df.columns:
                    self.raw_df['Subject'] = self.raw_df['ID']
                else:
                    self.logger.error("Subject column not found and cannot be inferred.")
                    raise ValueError("Subject column ('Subject', 'Participant', 'ID') not found in CSV.")
            self.raw_df['Subject'] = self.raw_df['Subject'].astype(str)
            self.logger.info(f"Data loaded successfully. Shape: {self.raw_df.shape}")
        except FileNotFoundError:
            self.logger.error(f"CSV file not found at {self.csv_filepath}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise

    def preprocess_labels(self):
        if self.raw_df is None:
            self.logger.error("Data not loaded. Call load_data() first.")
            raise ValueError("Data not loaded.")
        
        self.logger.info("Preprocessing labels (factorizing 'Class' column)")
        self.processed_df = self.raw_df.copy()
        self.processed_df['Class'] = self.processed_df['Class'].astype(str)

        self.processed_df["Class_original"] = self.processed_df["Class"]

        label_encoder = LabelEncoder()
        
        self.processed_df["Class"] = label_encoder.fit_transform(self.processed_df["Class"])
        self.class_labels = label_encoder.classes_
        self.num_classes = len(self.class_labels)
        self.logger.info(f"Factorized 'Class' into {self.num_classes} classes.")
        self.logger.info(f"Class mapping: {list(enumerate(self.class_labels))}")


    def _generate_windows_for_subject(self, subject_df, window_samples, overlap_samples):



        
        feature_columns = [col for col in subject_df.columns if col not in ['Subject', 'Class', 'Class_original']]
        X_data = subject_df[feature_columns].values
        Y_data = subject_df["Class"].values

        X_windowed, y_windowed = [], []
        

        stride = window_samples - overlap_samples
        if stride <= 0:
            self.logger.warning(f"Stride ({stride}) is not positive. Windowing may produce unexpected results or errors.")
            stride = 1

        for i in range(0, len(X_data) - window_samples + 1, stride):
            window_x = X_data[i : i + window_samples]
            window_y_segment = Y_data[i : i + window_samples]
            
            X_windowed.append(window_x)
            y_windowed.append(calculate_mode(window_y_segment))
        
        if not X_windowed:
            return np.array([]).reshape(0, window_samples, X_data.shape[1] if X_data.ndim > 1 else 0), np.array([])

        return np.array(X_windowed), np.array(y_windowed)

    def apply_windowing_to_subjects(self, subjects_list, window_samples, overlap_samples):
        if self.processed_df is None:
            self.logger.error("Data not preprocessed. Call preprocess_labels() first.")
            raise ValueError("Data not preprocessed.")

        all_X_windowed, all_y_windowed = [], []
        self.logger.info(f"Applying windowing: {window_samples} samples, {overlap_samples} overlap ({((overlap_samples/window_samples)*100):.1f}%).")

        for subject_id in np.unique(subjects_list):
            subject_df = self.processed_df[self.processed_df["Subject"] == subject_id]
            if subject_df.empty:
                self.logger.warning(f"No data found for subject {subject_id}. Skipping.")
                continue

            X_subj, y_subj = self._generate_windows_for_subject(subject_df, window_samples, overlap_samples)
            
            if X_subj.size > 0:
                all_X_windowed.append(X_subj)
                all_y_windowed.append(y_subj)
                self.logger.info(f"Subject {subject_id}: Generated {X_subj.shape[0]} windows of shape {X_subj.shape[1:]}")
            else:
                self.logger.warning(f"Subject {subject_id}: No windows generated.")
        
        if not all_X_windowed:
            self.logger.warning("No windows generated for any specified subjects.")

            feature_cols = [col for col in self.processed_df.columns if col not in ['Subject', 'Class', 'Class_original']]
            num_features = len(feature_cols)
            return np.array([]).reshape(0, window_samples, num_features), np.array([])


        final_X = np.vstack(all_X_windowed)
        final_y = np.hstack(all_y_windowed).reshape(-1, 1)
        



        indices = np.arange(final_X.shape[0])
        np.random.shuffle(indices)
        final_X = final_X[indices]
        final_y = final_y[indices]

        self.logger.info(f"Total windows generated for {len(subjects_list)} subjects: {final_X.shape[0]}")
        return final_X, final_y

    def get_data_splits(self, train_subjects, valid_subjects, test_subjects, 
                        window_duration_seconds, overlap_ratio, fs):
        
        window_samples = int(window_duration_seconds * fs)
        overlap_samples = int(window_samples * overlap_ratio)

        self.logger.info(f"Preparing data splits using window_samples={window_samples}, overlap_samples={overlap_samples}")

        X_train, y_train = self.apply_windowing_to_subjects(train_subjects, window_samples, overlap_samples)
        X_valid, y_valid = self.apply_windowing_to_subjects(valid_subjects, window_samples, overlap_samples)
        X_test, y_test = self.apply_windowing_to_subjects(test_subjects, window_samples, overlap_samples)
        

        X_train = np.nan_to_num(X_train.astype(float), nan=0.0)
        X_valid = np.nan_to_num(X_valid.astype(float), nan=0.0)
        X_test = np.nan_to_num(X_test.astype(float), nan=0.0)

        y_train = y_train.astype(int)
        y_valid = y_valid.astype(int)
        y_test = y_test.astype(int)
        
        return {
            "X_time_train": X_train, "y_train": y_train,
            "X_time_valid": X_valid, "y_valid": y_valid,
            "X_time_test": X_test, "y_test": y_test,
            "num_classes": self.num_classes,
            "class_labels": self.class_labels,
            "window_samples": window_samples
        }

    def generate_and_cache_datasets(self, window_size_options, overlap_ratio_options, fs, cache_dir, train_subjects, valid_subjects, test_subjects):
        self.logger.info(f"--- Starting Pre-generation and Caching of Datasets into '{cache_dir}'  ---")
        os.makedirs(cache_dir, exist_ok=True)
        if self.processed_df is None:
            self.logger.error("Data not preprocessed. Call preprocess_labels() first.")
            raise ValueError("Data not preprocessed.")

        for win_size_s in window_size_options:
            for overlap_r in overlap_ratio_options:
                window_samples = int(win_size_s * fs)
                overlap_samples = int(window_samples * overlap_r)

                filename = f"data_ws_{win_size_s}s_ol_{overlap_r}.npz"
                filepath = os.path.join(cache_dir, filename)

                if os.path.exists(filepath):
                    self.logger.info(f"Dataset already exists, skipping: {filename}")
                    continue

                self.logger.info(f"Generating dataset for ws={win_size_s}s, ol={overlap_r}...")


                X_train, y_train = self.apply_windowing_to_subjects(train_subjects, window_samples, overlap_samples)
                X_valid, y_valid = self.apply_windowing_to_subjects(valid_subjects, window_samples, overlap_samples)
                X_test, y_test = self.apply_windowing_to_subjects(test_subjects, window_samples, overlap_samples)


                n_train = X_train.shape[0]
                dev_ratio = self.config.ga.DEV_RATIO
                n_dev = int(n_train * dev_ratio)

                if n_dev == 0 and n_train > 0:
                    n_dev = 1


                np.random.seed(42)
                dev_indices = np.random.choice(n_train, size=n_dev, replace=False) if n_train > 0 else []

                X_time_dev = X_train[dev_indices]
                y_dev = y_train[dev_indices]
                self.logger.info(f"Created a fixed dev set with {X_time_dev.shape[0]} samples.")



                feature_extractor = FeatureExtractor(self.config, self.logger)
                nperseg = window_samples // self.config.data_preprocessing.INITIAL_NPERSEG_DIVISOR

                X_freq_train = feature_extractor.compute_stft_features(X_train, fs, nperseg)
                X_freq_valid = feature_extractor.compute_stft_features(X_valid, fs, nperseg)
                X_freq_test = feature_extractor.compute_stft_features(X_test, fs, nperseg)
                X_freq_dev = feature_extractor.compute_stft_features(X_time_dev, fs, nperseg)


                np.savez_compressed(
                    filepath,
                    X_time_train=np.nan_to_num(X_train.astype(float), nan=0.0), y_train=y_train.astype(int),
                    X_time_valid=np.nan_to_num(X_valid.astype(float), nan=0.0), y_valid=y_valid.astype(int),
                    X_time_test=np.nan_to_num(X_test.astype(float), nan=0.0), y_test=y_test.astype(int),
                    X_time_dev=np.nan_to_num(X_time_dev.astype(float), nan=0.0), y_dev=y_dev.astype(int),
                    X_freq_train=X_freq_train, X_freq_valid=X_freq_valid, X_freq_test=X_freq_test,
                    X_freq_dev=X_freq_dev,
                    num_classes=self.num_classes,
                    class_labels=self.class_labels,
                    window_samples=window_samples
                )
                self.logger.info(f"Saved dataset to {filepath}")

        self.logger.info("--- Finished Caching All Datasets ---")
