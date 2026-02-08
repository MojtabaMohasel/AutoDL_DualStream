
# 🧠 Automated Dual-Stream Deep Network Design for Activity Recognition

This repository provides a complete framework for **automated optimization and training of dual-stream deep learning models** for time-series sensor data (e.g., IMU signals). It integrates **Genetic Algorithms (GA)** with deep neural networks to automatically discover optimal:

-   Data windowing parameters
    
-   Feature extraction settings
    
-   Network architectures
    
-   Training hyperparameters
    

The system supports **time-domain and frequency-domain fusion** using CNN and LSTM architectures.

----------

## 📌 Key Features

-   ✅ Automatic data preprocessing and windowing
    
-   ✅ STFT-based frequency feature extraction
    
-   ✅ Dual-stream CNN/LSTM architecture
    
-   ✅ Genetic Algorithm optimization
    
-   ✅ Class imbalance handling (Focal Loss / Weighted CE)
    
-   ✅ Early stopping and time-based stopping
    
-   ✅ Final model retraining and evaluation
    
-   ✅ Experiment logging and checkpointing
    

----------

## 📂 Project Structure

```
.
├── data_handler.py        # Data loading, label processing, windowing
├── data_loader.py         # GA checkpoint loading utilities
├── feature_extractor.py   # Frequency-domain feature extraction (STFT)
├── genetic_optimizer.py   # Genetic Algorithm implementation
├── model_architectures.py # Dynamic dual-stream model builder
├── trainer.py             # Training, validation, evaluation
├── utils.py               # Logging, callbacks, helpers
├── dual-stream.ipynb      # Example notebook
└── README.md

```

----------

## ⚙️ Requirements

### Python Version

```
Python >= 3.8

```

### Dependencies

Install required packages:

```bash
pip install numpy pandas scipy scikit-learn tensorflow typeguard

```

Optional (recommended):

```bash
pip install matplotlib seaborn jupyter

```

For GPU support:

```bash
pip install tensorflow-gpu

```

----------

## 📊 Dataset Format

Input data must be provided as a CSV file.

### Required Columns

Column Name

Description

Subject / Participant / ID

Subject identifier

Class / LABEL

Activity label

Other columns

Sensor channels (IMU, etc.)

### Example

```csv
Subject,AccX,AccY,AccZ,GyroX,GyroY,GyroZ,Class
1,0.12,0.45,0.32,1.01,0.87,0.55,Walking
1,0.14,0.47,0.29,1.05,0.89,0.58,Walking

```

The framework automatically:

-   Handles missing values
    
-   Infers subject IDs
    
-   Encodes class labels
    
-   Normalizes formats
    

----------

## 🚀 System Pipeline

```
Raw CSV Data
     ↓
Windowing & Label Processing
     ↓
STFT Feature Extraction
     ↓
Cached Datasets (.npz)
     ↓
Genetic Algorithm Optimization
     ↓
Model Selection
     ↓
Final Training
     ↓
Evaluation & Saving

```

----------

## 🛠 Step 1: Data Preprocessing and Caching

Before running optimization, datasets must be generated and cached.

### Example

```python
from data_handler import DataHandler
from utils import setup_logger

logger = setup_logger("runs", "preprocess")

handler = DataHandler("data.csv", config, logger)

handler.load_data()
handler.preprocess_labels()

handler.generate_and_cache_datasets(
    window_size_options=[1.5, 2.0],
    overlap_ratio_options=[0.5, 0.75],
    fs=100,
    cache_dir="cached_data",
    train_subjects=train_ids,
    valid_subjects=valid_ids,
    test_subjects=test_ids
)

```

Generated files:

```
cached_data/
 └── data_ws_1.5s_ol_0.5.npz

```

----------

## 🧬 Step 2: Genetic Algorithm Optimization

Initialize and run the GA to search for optimal models.

### Example

```python
from genetic_optimizer import GeneticAlgorithm, ChromosomeHelper

helper = ChromosomeHelper(num_classes, config, logger)

ga = GeneticAlgorithm(
    chromosome_helper=helper,
    config=config,
    logger=logger,
    run_dir="runs/experiment_1"
)

ga.initialize_population()
ga.run()

```

The optimizer automatically:

-   Builds candidate models
    
-   Trains and evaluates them
    
-   Applies mutation and crossover
    
-   Saves checkpoints
    

----------

## 🧠 Step 3: Final Model Retraining

After optimization, retrain the best model using combined training and validation data.

### Example

```python
from trainer import retrain_and_evaluate_best_model

retrain_and_evaluate_best_model(
    best_individual_parts,
    best_fitness,
    config,
    logger,
    run_dir="runs/experiment_1",
    epochs_retrain=50
)

```

Outputs:

-   Final trained model
    
-   Test classification report
    
-   Confusion matrices
    
-   Saved model parameters
    

----------

## 🏗 Model Architecture

The framework supports flexible multi-stream architectures.

### Available Streams

Stream

Description

CNN-1D

Time-domain convolution

CNN-2D

Frequency-domain convolution

LSTM

Temporal modeling

MLP

Shared classification head

### Stream Configuration

Edit in `config.py`:

```python
ACTIVE_STREAMS = {
    "cnn_1d": True,
    "cnn_2d": True,
    "lstm": False
}

```

----------

## 📈 Training & Evaluation Metrics

The system reports:

-   Accuracy
    
-   Macro F1-score
    
-   Precision / Recall
    
-   Confusion Matrix
    
-   ROC-AUC (macro)
    

Reports are saved as JSON files.

----------

## 📁 Output Directory Structure

Each experiment produces:

```
runs/experiment_1/
 ├── preprocess.log
 ├── ga.log
 ├── config.json
 ├── final_test_results.json
 ├── best_model_*.keras
 └── checkpoints/

```

----------

## ⚠️ Troubleshooting

### No Windows Generated

-   Check window size
    
-   Check sampling rate
    
-   Verify signal length
    

### Empty Validation/Test Sets

-   Verify subject splits
    
-   Ensure sufficient samples
    

### Slow Training

-   Enable GPU
    
-   Reduce population size
    
-   Reduce epochs
    

### CUDA Errors

Install compatible drivers and TensorFlow version.

----------

## 📓 Example Usage

An example workflow is provided in:

```
dual-stream.ipynb

```

It demonstrates:

-   Data loading
    
-   Feature extraction
    
-   Training
    
-   Evaluation
    

----------

## 📚 Citation

If you use this framework in academic work, please cite:

```bibtex
@misc{dualstream2026,
  title={Automated Dual-Stream Deep Network Design for Activity Recognition},
  author={Seyed Mojtaba Mohasel, Alireza Afzal Aghaei, John Sheppard, Corey Pew},
  year={2026}
}

```

----------

## 👤 Author

Developed by  Seyed Mojtaba Mohasel, Alireza Afzal Aghaei






