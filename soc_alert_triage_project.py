#!/usr/bin/env python3
"""
SOC Automated Alert Triage using Machine Learning
=================================================

This script is designed for Google Colab, Jupyter Notebook, or a local Python environment.

Project focus:
- Automated Security Operations Centre alert triage
- False positive reduction
- SOC analyst workload reduction
- Models: Random Forest, XGBoost, and optional LSTM
- Datasets: UNSW-NB15, CICIDS 2017, CICIDS 2018, NSL-KDD, or any labelled cybersecurity CSV

Quick Colab use:
1. Upload kaggle.json to Colab if you want automatic Kaggle download.
2. Run:
   !python soc_alert_triage_project.py --download unsw --target label --skip-lstm

For your own CSV:
   !python soc_alert_triage_project.py --dataset-path /content/my_dataset.csv --target Label --skip-lstm

Outputs:
- outputs/model_comparison_results.csv
- outputs/soc_triage_predictions.csv
- outputs/confusion_matrix_*.png
- outputs/roc_curve_*.png
- outputs/feature_importance_*.png
- outputs/*.joblib saved models
"""

import argparse
import os
import sys
import json
import glob
import shutil
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# -----------------------------
# Package installation utilities
# -----------------------------

def pip_install(package: str) -> None:
    """Install a Python package quietly if needed."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def ensure_packages(skip_lstm: bool = False) -> None:
    """Install required packages if they are missing."""
    required = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("joblib", "joblib"),
        ("xgboost", "xgboost"),
    ]

    if not skip_lstm:
        required.append(("tensorflow", "tensorflow"))

    for import_name, package_name in required:
        try:
            __import__(import_name)
        except ImportError:
            print(f"Installing missing package: {package_name}")
            pip_install(package_name)


# -----------------------------
# Dataset download utilities
# -----------------------------

KAGGLE_DATASETS = {
    "unsw": {
        "slug": "mrwellsdavid/unsw-nb15",
        "folder": "datasets/unsw_nb15",
        "target_hint": "label",
        "notes": "Best starting dataset. Contains UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in many Kaggle mirrors."
    },
    "cicids2017": {
        "slug": "chethuhn/network-intrusion-dataset",
        "folder": "datasets/cicids2017",
        "target_hint": "Label",
        "notes": "CICIDS 2017 mirror. Usually contains multiple CSV files with Label column."
    },
    "cicids2018": {
        "slug": "solarmainframe/ids-intrusion-csv",
        "folder": "datasets/cicids2018",
        "target_hint": "Label",
        "notes": "CICIDS 2018 mirror. Large files may require high RAM."
    },
    "nslkdd": {
        "slug": "hassan06/nslkdd",
        "folder": "datasets/nsl_kdd",
        "target_hint": "class",
        "notes": "NSL-KDD benchmark mirror. Some files may not have headers, so manual column handling can be needed."
    }
}


def setup_kaggle_credentials(kaggle_json_path: str = "kaggle.json") -> bool:
    """
    Configure Kaggle credentials in Colab or local environment.
    Returns True if credentials are available.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    destination = kaggle_dir / "kaggle.json"

    if Path(kaggle_json_path).exists():
        shutil.copy(kaggle_json_path, destination)
        os.chmod(destination, 0o600)
        return True

    if destination.exists():
        os.chmod(destination, 0o600)
        return True

    return False


def download_kaggle_dataset(dataset_key: str) -> Path:
    """
    Download a dataset from Kaggle.
    Requires kaggle.json credentials.
    """
    if dataset_key not in KAGGLE_DATASETS:
        valid = ", ".join(KAGGLE_DATASETS.keys())
        raise ValueError(f"Unknown dataset key '{dataset_key}'. Valid options: {valid}")

    try:
        __import__("kaggle")
    except ImportError:
        print("Installing Kaggle package...")
        pip_install("kaggle")

    has_credentials = setup_kaggle_credentials()
    if not has_credentials:
        raise RuntimeError(
            "Kaggle credentials were not found. In Colab, upload kaggle.json first:\n"
            "from google.colab import files\n"
            "files.upload()\n"
            "Then run this script again."
        )

    config = KAGGLE_DATASETS[dataset_key]
    output_folder = Path(config["folder"])
    output_folder.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", config["slug"],
        "-p", str(output_folder),
        "--unzip"
    ]

    print(f"Downloading dataset: {dataset_key}")
    print(f"Kaggle slug: {config['slug']}")
    subprocess.check_call(command)
    return output_folder


def find_csv_files(folder: Path) -> list:
    """Find all CSV files in a folder recursively."""
    return sorted([Path(p) for p in glob.glob(str(folder / "**" / "*.csv"), recursive=True)])


# -----------------------------
# Data loading and preprocessing
# -----------------------------

def load_csv_safely(path: Path, max_rows: int = None):
    """Load a CSV with common encodings."""
    import pandas as pd

    encodings = ["utf-8", "latin1", "ISO-8859-1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, nrows=max_rows, low_memory=False)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not load CSV file: {path}. Last error: {last_error}")


def load_dataset_from_path(dataset_path: str, sample_rows: int = None):
    """
    Load a CSV file or combine multiple CSVs from a folder.
    If folder contains UNSW train/test files, combine them.
    """
    import pandas as pd

    path = Path(dataset_path)

    if path.is_file():
        print(f"Loading CSV file: {path}")
        return load_csv_safely(path, max_rows=sample_rows)

    if path.is_dir():
        csv_files = find_csv_files(path)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found inside: {path}")

        print("CSV files found:")
        for f in csv_files[:20]:
            print(f" - {f}")
        if len(csv_files) > 20:
            print(f"... and {len(csv_files) - 20} more CSV files.")

        # Prefer UNSW train and test if present
        unsw_files = [f for f in csv_files if "UNSW_NB15" in f.name or "UNSW" in f.name]
        if unsw_files:
            csv_files_to_load = unsw_files
        else:
            csv_files_to_load = csv_files

        frames = []
        for file in csv_files_to_load:
            try:
                df_part = load_csv_safely(file, max_rows=sample_rows)
                df_part["source_file"] = file.name
                frames.append(df_part)
                print(f"Loaded {file.name}: {df_part.shape}")
            except Exception as exc:
                print(f"Skipped {file}: {exc}")

        if not frames:
            raise RuntimeError("No CSV files could be loaded.")

        df = pd.concat(frames, ignore_index=True)

        if sample_rows and len(df) > sample_rows:
            df = df.sample(n=sample_rows, random_state=42).reset_index(drop=True)

        return df

    raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")


def clean_column_names(df):
    """Strip whitespace and standardise problematic column names."""
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def detect_target_column(df, preferred_target: str = None) -> str:
    """Detect a likely target column."""
    if preferred_target:
        preferred_clean = preferred_target.strip().replace(" ", "_")
        if preferred_clean in df.columns:
            return preferred_clean

    candidates = [
        "label", "Label", "LABEL",
        "attack_cat", "Attack", "attack", "class", "Class",
        "outcome", "target", "Target", "Category", "category"
    ]

    for col in candidates:
        clean_col = col.strip().replace(" ", "_")
        if clean_col in df.columns:
            return clean_col

    lower_map = {c.lower(): c for c in df.columns}
    for key in ["label", "attack_cat", "class", "target", "outcome"]:
        if key in lower_map:
            return lower_map[key]

    raise ValueError(
        "Could not detect target column. Please pass --target with the correct label column name.\n"
        f"Available columns: {list(df.columns)[:80]}"
    )


def prepare_features_and_target(df, target_column: str):
    """
    Prepare X and y.
    Drops obvious ID-like columns.
    Encodes target labels to integers.
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = clean_column_names(df)
    target_column = target_column.strip().replace(" ", "_")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with missing target
    df = df.dropna(subset=[target_column]).reset_index(drop=True)

    # Drop columns that are usually identifiers or leakage-prone only if present
    drop_like = [
        "id", "ID", "Flow_ID", "Timestamp", "timestamp",
        "Src_IP", "Dst_IP", "Source_IP", "Destination_IP",
        "source_file"
    ]

    columns_to_drop = []
    for col in df.columns:
        if col == target_column:
            continue
        if col in drop_like:
            columns_to_drop.append(col)

    X = df.drop(columns=[target_column] + columns_to_drop, errors="ignore")
    y_raw = df[target_column].astype(str).str.strip()

    # Convert common benign names consistently
    y_raw = y_raw.replace({
        "BENIGN": "Benign",
        "Normal": "Benign",
        "normal": "Benign",
        "0": "Benign" if target_column.lower() == "label" else "0",
        "1": "Attack" if target_column.lower() == "label" else "1",
    })

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)

    return X, y, y_raw, target_encoder


def build_preprocessor(X):
    """Create preprocessing pipeline for numeric and categorical features."""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features


# -----------------------------
# Model training and evaluation
# -----------------------------

def evaluate_model(model_name, model, X_test, y_test, target_encoder, output_dir):
    """Evaluate a trained sklearn-compatible model and save reports/figures."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score, roc_curve
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None

    average_type = "binary" if len(target_encoder.classes_) == 2 else "weighted"

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average_type, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average_type, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average=average_type, zero_division=0),
    }

    if y_proba is not None:
        try:
            if len(target_encoder.classes_) == 2:
                results["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                results["roc_auc"] = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            results["roc_auc"] = None
    else:
        results["roc_auc"] = None

    # False positive rate for binary case
    if len(target_encoder.classes_) == 2:
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            results["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
            results["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        results["false_positive_rate"] = None
        results["false_negative_rate"] = None

    # Save classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=[str(c) for c in target_encoder.classes_],
        zero_division=0
    )
    with open(output_dir / f"classification_report_{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    tick_marks = np.arange(len(target_encoder.classes_))
    plt.xticks(tick_marks, target_encoder.classes_, rotation=45, ha="right")
    plt.yticks(tick_marks, target_encoder.classes_)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=300)
    plt.close()

    # ROC curve for binary classification
    if y_proba is not None and len(target_encoder.classes_) == 2:
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"{model_name}")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
            plt.title(f"ROC Curve - {model_name}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"roc_curve_{model_name}.png", dpi=300)
            plt.close()
        except Exception:
            pass

    return results, y_pred, y_proba


def train_random_forest(X_train, y_train, preprocessor):
    """Train Random Forest model."""
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, preprocessor, num_classes):
    """Train XGBoost model."""
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    if num_classes == 2:
        objective = "binary:logistic"
        eval_metric = "logloss"
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective=objective,
            eval_metric=eval_metric,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    return model


def train_lstm(X_train, y_train, X_test, y_test, preprocessor, num_classes, output_dir, epochs=5):
    """
    Train a simple LSTM model.

    Note:
    This treats each transformed feature vector as a sequence.
    For a stronger dissertation implementation, use real timestamped alert sequences
    if the dataset supports it.
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping

    output_dir = Path(output_dir)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse to dense if needed
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    X_train_seq = X_train_processed.reshape((X_train_processed.shape[0], X_train_processed.shape[1], 1))
    X_test_seq = X_test_processed.reshape((X_test_processed.shape[0], X_test_processed.shape[1], 1))

    if num_classes == 2:
        y_train_final = y_train
        y_test_final = y_test
        output_units = 1
        activation = "sigmoid"
        loss = "binary_crossentropy"
    else:
        y_train_final = to_categorical(y_train, num_classes=num_classes)
        y_test_final = to_categorical(y_test, num_classes=num_classes)
        output_units = num_classes
        activation = "softmax"
        loss = "categorical_crossentropy"

    model = Sequential([
        LSTM(64, input_shape=(X_train_seq.shape[1], 1)),
        Dropout(0.30),
        Dense(32, activation="relu"),
        Dropout(0.20),
        Dense(output_units, activation=activation)
    ])

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    model.fit(
        X_train_seq,
        y_train_final,
        validation_split=0.2,
        epochs=epochs,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model and preprocessor
    model.save(output_dir / "lstm_model.keras")

    if num_classes == 2:
        pred_proba = model.predict(X_test_seq).reshape(-1)
        y_pred = (pred_proba >= 0.5).astype(int)
        y_proba = np.column_stack([1 - pred_proba, pred_proba])
    else:
        y_proba = model.predict(X_test_seq)
        y_pred = np.argmax(y_proba, axis=1)

    return model, y_pred, y_proba


def evaluate_lstm_predictions(y_test, y_pred, y_proba, target_encoder, output_dir):
    """Evaluate LSTM predictions."""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score, roc_curve
    )

    model_name = "LSTM"
    output_dir = Path(output_dir)

    average_type = "binary" if len(target_encoder.classes_) == 2 else "weighted"

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average_type, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average_type, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average=average_type, zero_division=0),
    }

    try:
        if len(target_encoder.classes_) == 2:
            results["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            results["roc_auc"] = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        results["roc_auc"] = None

    if len(target_encoder.classes_) == 2:
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            results["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
            results["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        results["false_positive_rate"] = None
        results["false_negative_rate"] = None

    report = classification_report(
        y_test,
        y_pred,
        target_names=[str(c) for c in target_encoder.classes_],
        zero_division=0
    )
    with open(output_dir / "classification_report_LSTM.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix - LSTM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    tick_marks = np.arange(len(target_encoder.classes_))
    plt.xticks(tick_marks, target_encoder.classes_, rotation=45, ha="right")
    plt.yticks(tick_marks, target_encoder.classes_)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_LSTM.png", dpi=300)
    plt.close()

    if len(target_encoder.classes_) == 2:
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label="LSTM")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
            plt.title("ROC Curve - LSTM")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "roc_curve_LSTM.png", dpi=300)
            plt.close()
        except Exception:
            pass

    return results


# -----------------------------
# SOC triage and MITRE context
# -----------------------------

def map_to_mitre_context(label: str) -> str:
    """Simple MITRE ATT&CK style contextual mapping based on predicted label text."""
    text = str(label).lower()

    if any(k in text for k in ["benign", "normal"]):
        return "No active threat mapping"
    if any(k in text for k in ["dos", "ddos", "heartbleed"]):
        return "Impact: Denial of Service related behaviour"
    if any(k in text for k in ["brute", "ftp", "ssh", "password"]):
        return "Credential Access: Brute force or password attack behaviour"
    if any(k in text for k in ["web", "sql", "xss", "injection"]):
        return "Initial Access or Execution: Web application attack behaviour"
    if any(k in text for k in ["bot", "botnet"]):
        return "Command and Control: Botnet style behaviour"
    if any(k in text for k in ["infiltration", "backdoor", "shellcode"]):
        return "Persistence or Lateral Movement: Possible internal compromise behaviour"
    if any(k in text for k in ["recon", "scan", "analysis", "fuzzers"]):
        return "Discovery or Reconnaissance: Scanning or probing behaviour"
    if any(k in text for k in ["exploits", "generic", "worms"]):
        return "Execution or Exploitation: Exploit based attack behaviour"

    return "Requires analyst review for MITRE ATT&CK mapping"


def assign_triage_priority(predicted_label: str, confidence: float) -> str:
    """Assign SOC triage priority from model prediction and confidence."""
    label_text = str(predicted_label).lower()

    if any(k in label_text for k in ["benign", "normal"]):
        return "Likely false positive or benign"

    if confidence >= 0.85:
        return "High priority threat"
    if confidence >= 0.65:
        return "Medium priority alert"
    if confidence >= 0.50:
        return "Low priority alert"
    return "Low confidence manual review"


def save_triage_predictions(model, model_name, X_test, y_test, target_encoder, output_dir, max_rows=1000):
    """Save SOC-style triage output for analyst use."""
    import numpy as np
    import pandas as pd

    output_dir = Path(output_dir)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        confidence = y_proba.max(axis=1)
    else:
        confidence = np.ones(len(y_pred))

    pred_labels = target_encoder.inverse_transform(y_pred)
    actual_labels = target_encoder.inverse_transform(y_test)

    triage_df = pd.DataFrame({
        "actual_label": actual_labels,
        "predicted_label": pred_labels,
        "model_confidence": confidence,
        "triage_priority": [
            assign_triage_priority(label, conf) for label, conf in zip(pred_labels, confidence)
        ],
        "mitre_context": [
            map_to_mitre_context(label) for label in pred_labels
        ]
    })

    triage_df = triage_df.head(max_rows)
    triage_df.to_csv(output_dir / f"soc_triage_predictions_{model_name}.csv", index=False)

    return triage_df


def save_feature_importance(model, model_name, preprocessor, output_dir, top_n=20):
    """Save feature importance plot for tree-based pipeline models."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    if not hasattr(model.named_steps["classifier"], "feature_importances_"):
        return

    try:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        importances = model.named_steps["classifier"].feature_importances_

        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).head(top_n)

        df_imp.to_csv(output_dir / f"feature_importance_{model_name}.csv", index=False)

        plt.figure(figsize=(10, 7))
        plt.barh(df_imp["feature"][::-1], df_imp["importance"][::-1])
        plt.title(f"Top Feature Importance - {model_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(output_dir / f"feature_importance_{model_name}.png", dpi=300)
        plt.close()
    except Exception as exc:
        print(f"Could not save feature importance for {model_name}: {exc}")


# -----------------------------
# Main workflow
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="SOC Alert Triage ML Project Script")

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to CSV file or folder containing CSV files."
    )

    parser.add_argument(
        "--download",
        type=str,
        default=None,
        choices=list(KAGGLE_DATASETS.keys()),
        help="Download dataset from Kaggle. Options: unsw, cicids2017, cicids2018, nslkdd."
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target/label column name. Example: label or Label."
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=80000,
        help="Maximum rows to use. Use smaller value for Colab RAM. Set 0 for all rows."
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split size."
    )

    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM to save time and memory."
    )

    parser.add_argument(
        "--lstm-epochs",
        type=int,
        default=5,
        help="Number of LSTM training epochs."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Folder where outputs will be saved."
    )

    args = parser.parse_args()

    ensure_packages(skip_lstm=args.skip_lstm)

    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = None if args.sample == 0 else args.sample

    # Download or load dataset
    if args.download:
        dataset_folder = download_kaggle_dataset(args.download)
        dataset_path = str(dataset_folder)
        if args.target is None:
            args.target = KAGGLE_DATASETS[args.download]["target_hint"]
    elif args.dataset_path:
        dataset_path = args.dataset_path
    else:
        raise ValueError(
            "Please provide either --dataset-path or --download.\n"
            "Example: python soc_alert_triage_project.py --download unsw --target label --skip-lstm"
        )

    print("\nLoading dataset...")
    df = load_dataset_from_path(dataset_path, sample_rows=sample_rows)
    df = clean_column_names(df)

    print(f"Dataset shape: {df.shape}")
    print("First columns:", list(df.columns)[:30])

    target_column = detect_target_column(df, args.target)
    print(f"Detected target column: {target_column}")

    X, y, y_raw, target_encoder = prepare_features_and_target(df, target_column)
    print(f"Feature shape: {X.shape}")
    print("Target classes:", list(target_encoder.classes_))

    class_counts = pd.Series(y_raw).value_counts()
    print("\nClass distribution:")
    print(class_counts)
    class_counts.to_csv(output_dir / "class_distribution.csv")

    # Split data
    stratify_arg = y if len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify_arg
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

    print(f"\nNumeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    all_results = []

    # Random Forest
    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train, preprocessor)
    rf_results, _, _ = evaluate_model("Random_Forest", rf_model, X_test, y_test, target_encoder, output_dir)
    all_results.append(rf_results)
    joblib.dump(rf_model, output_dir / "random_forest_pipeline.joblib")
    save_feature_importance(rf_model, "Random_Forest", preprocessor, output_dir)
    save_triage_predictions(rf_model, "Random_Forest", X_test, y_test, target_encoder, output_dir)
    print("Random Forest results:", rf_results)

    # XGBoost
    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, preprocessor, num_classes=len(target_encoder.classes_))
    xgb_results, _, _ = evaluate_model("XGBoost", xgb_model, X_test, y_test, target_encoder, output_dir)
    all_results.append(xgb_results)
    joblib.dump(xgb_model, output_dir / "xgboost_pipeline.joblib")
    save_feature_importance(xgb_model, "XGBoost", preprocessor, output_dir)
    save_triage_predictions(xgb_model, "XGBoost", X_test, y_test, target_encoder, output_dir)
    print("XGBoost results:", xgb_results)

    # LSTM
    if not args.skip_lstm:
        print("\nTraining LSTM...")
        lstm_preprocessor, _, _ = build_preprocessor(X_train)
        lstm_model, lstm_pred, lstm_proba = train_lstm(
            X_train,
            y_train,
            X_test,
            y_test,
            lstm_preprocessor,
            num_classes=len(target_encoder.classes_),
            output_dir=output_dir,
            epochs=args.lstm_epochs
        )
        lstm_results = evaluate_lstm_predictions(y_test, lstm_pred, lstm_proba, target_encoder, output_dir)
        all_results.append(lstm_results)
        joblib.dump(lstm_preprocessor, output_dir / "lstm_preprocessor.joblib")
        print("LSTM results:", lstm_results)
    else:
        print("\nSkipping LSTM as requested.")

    # Save model comparison
    results_df = pd.DataFrame(all_results).sort_values("f1_score", ascending=False)
    results_df.to_csv(output_dir / "model_comparison_results.csv", index=False)

    # Save target encoder
    joblib.dump(target_encoder, output_dir / "target_label_encoder.joblib")

    # Save project metadata
    metadata = {
        "dataset_path": dataset_path,
        "target_column": target_column,
        "classes": [str(c) for c in target_encoder.classes_],
        "rows_used": int(len(df)),
        "features_used": int(X.shape[1]),
        "numeric_features": len(numeric_features),
        "categorical_features": len(categorical_features),
        "best_model_by_f1": str(results_df.iloc[0]["model"]) if len(results_df) else None
    }

    with open(output_dir / "project_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")
    print(results_df)

    print("\nOutputs saved in:", output_dir.resolve())
    print("Important files:")
    print(" - model_comparison_results.csv")
    print(" - soc_triage_predictions_Random_Forest.csv")
    print(" - soc_triage_predictions_XGBoost.csv")
    print(" - confusion_matrix_*.png")
    print(" - classification_report_*.txt")
    print(" - random_forest_pipeline.joblib")
    print(" - xgboost_pipeline.joblib")


if __name__ == "__main__":
    main()
