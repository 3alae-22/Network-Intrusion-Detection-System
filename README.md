# Network Intrusion Detection System

A machine learning project for detecting network intrusions using the KDD Cup 1999 dataset. This system uses a two-stage approach: first detecting if traffic is normal or an attack, then identifying the specific type of attack.

## About

This project builds a Network Intrusion Detection System (NIDS) using machine learning. It analyzes network traffic patterns to detect cyber attacks in real-time.

**Key Features:**
- Binary classification (Normal vs Attack)
- Multi-class attack type identification (23 different attack types)
- Real-time network monitoring simulation
- Support for both RandomForest and XGBoost algorithms
- Handles imbalanced data using SMOTE

## Dataset

The project uses the KDD Cup 1999 dataset:
- 494,020 network connections (145,584 unique after cleaning)
- 41 features describing network connection characteristics
- 23 different types of attacks grouped into 4 categories:
  - DoS (Denial of Service)
  - Probe (Network scanning)
  - R2L (Remote to Local attacks)
  - U2R (User to Root privilege escalation)

**Note:** The dataset is not included in this repository due to size. See setup instructions below.

## Project Structure

```
network-intrusion-detection/
├── DataAnalysis.ipynb                          # Data exploration
├── create_datasets.py                          # Split data into train/test
├── train_modelsRandomForest.py                 # Train with RandomForest
├── train_modelsXgBoostOptimise.py             # Train with XGBoost (optimized)
├── train_modelsXgBoostOptimise_RandomForest.py # Hybrid approach
├── evaluate_models.py                          # Test model performance
├── predict_app.py                              # Make predictions
├── live_monitor.py                             # Real-time monitoring demo
├── requirements.txt                            # Python packages needed
└── README.md                                   # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the Dataset

Download the KDD Cup 1999 dataset and save it as `kdd_df.csv` in the project folder.

**Download sources:**
- Official: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
- Kaggle: https://www.kaggle.com/datasets/hassan06/nslkdd
- UCI: https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data

### 3. Prepare the Data

```bash
python create_datasets.py
```

This creates `kdd_train.csv` and `kdd_test.csv` (80/20 split).

## Usage

### Train Models

Choose one of these training options:

**Option 1: RandomForest (faster)**
```bash
python train_modelsRandomForest.py
```

**Option 2: XGBoost with hyperparameter tuning (recommended)**
```bash
python train_modelsXgBoostOptimise.py
```

**Option 3: Hybrid (XGBoost for binary + RandomForest for multiclass)**
```bash
python train_modelsXgBoostOptimise_RandomForest.py
```

This will create 4 files:
- `binary_detector_pipeline.pkl` - Model to detect attacks
- `multiclass_classifier_pipeline.pkl` - Model to classify attack types
- `binary_label_encoder.pkl` - Label encoder for binary model
- `multiclass_label_encoder.pkl` - Label encoder for multiclass model

### Evaluate Performance

```bash
python evaluate_models.py
```

Shows accuracy, precision, recall, and F1-scores for both models.

### Make Predictions

```bash
python predict_app.py
```

Tests the models with sample data.

### Live Monitoring Demo

```bash
python live_monitor.py
```

Simulates real-time network monitoring and generates alerts for detected attacks.

## How It Works

### Two-Stage Detection

**Stage 1: Binary Detection**
- Input: Network connection features
- Output: "Normal" or "Attack"
- Algorithm: XGBoost or RandomForest

**Stage 2: Attack Classification**
- Input: Connections classified as attacks
- Output: Specific attack type (e.g., "dos", "probe", "u2r", "r2l")
- Algorithm: RandomForest or XGBoost

### Data Preprocessing

1. **Categorical features:** One-hot encoding for protocol_type, service, and flag
2. **Numeric features:** Standard scaling
3. **Skewed features:** Log transformation + standard scaling (19 features total)
4. **Class imbalance:** SMOTE oversampling

### Skewed Features
Features with high skewness (>5) that get log transformation:
- src_bytes, dst_bytes, duration
- hot, num_failed_logins, num_compromised
- num_root, num_file_creations, num_shells
- root_shell, num_access_files, is_guest_login
- land, wrong_fragment, srv_count
- dst_host_srv_diff_host_rate, diff_srv_rate, su_attempted

## Results

Add your results here after training the models.

Example metrics to track:
- Binary Classification Accuracy
- Attack Detection Rate (Recall)
- False Positive Rate
- Multiclass F1-Score

## Files Generated

During training and testing, these files are created:

**Models:**
- `binary_detector_pipeline.pkl`
- `multiclass_classifier_pipeline.pkl`
- `binary_label_encoder.pkl`
- `multiclass_label_encoder.pkl`

**Data:**
- `kdd_train.csv`
- `kdd_test.csv`

**Evaluation:**
- `confusion_matrix_binary.png`
- `confusion_matrix_multiclass.png`

**Note:** These files are excluded from Git (see .gitignore)

## Technologies Used

- Python 3.8+
- scikit-learn - Machine learning
- XGBoost - Gradient boosting
- pandas - Data manipulation
- numpy - Numerical operations
- imbalanced-learn - SMOTE for class imbalancing
- matplotlib/seaborn - Visualization

## Contributing

Feel free to fork this project and submit pull requests.

## License

MIT License - feel free to use this project for learning and development.

## Contact

Your Name - alaeaghoutane32@gmail.com

## Acknowledgments

- KDD Cup 1999 dataset providers
- scikit-learn and XGBoost communities
