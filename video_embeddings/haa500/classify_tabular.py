import os
import time
import glob
import torch
import numpy as np
import pandas as pd
import logging
import argparse
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def load_split_ids(split_file):
    with open(split_file, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def load_data(split_ids, embedding_folder):
    X, y = [], []
    for file_id in split_ids:
        file_path = os.path.join(embedding_folder, file_id + '.pt')
        if not os.path.exists(file_path):
            logging.warning(f"Missing file: {file_path}")
            continue
        vec = torch.load(file_path)
        X.append(vec.numpy())
        y.append(file_id.split('_')[0])
    return np.array(X), np.array(y)

def tune_model(model, param_grid, X_train, y_train, X_val, y_val, model_name):
    logging.info(f"Tuning {model_name}...")
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    val_acc = best_model.score(X_val, y_val)
    logging.info(f"Best {model_name} val accuracy: {val_acc:.4f}")
    return best_model, grid.best_params_, val_acc

def evaluate_and_log(model, X_test, y_test, output_path, model_name):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_path, f"{model_name}_report.csv"))
    logging.info(f"Saved evaluation metrics for {model_name}.")
    return accuracy_score(y_test, y_pred)

def build_nn(hidden_layer_sizes=(100,), alpha=0.0001, activation='relu'):
    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, activation=activation, max_iter=500)

def main(args):
    os.makedirs(args.output_path, exist_ok=True)

    train_ids = load_split_ids(os.path.join(args.splits_path, 'train.txt'))
    val_ids   = load_split_ids(os.path.join(args.splits_path, 'val.txt'))
    test_ids  = load_split_ids(os.path.join(args.splits_path, 'test.txt'))

    X_train, y_train = load_data(train_ids, args.input_path)
    X_val, y_val = load_data(val_ids, args.input_path)
    X_test, y_test = load_data(test_ids, args.input_path)

    logging.info(f"Training Shape: {X_train.shape, y_train.shape}, Validation Shape: {X_val.shape}, Test Shape: {X_test.shape}")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    results = []

    try:
        # Random Forest
        start = time.time()
        rf, rf_best_params, rf_val_acc = tune_model(RandomForestClassifier(), {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
        }, X_train, y_train, X_val, y_val, "RandomForest")
        train_time = time.time() - start

        start = time.time()
        rf_test_acc = evaluate_and_log(rf, X_test, y_test, args.output_path, "RandomForest")
        test_time = time.time() - start

        with open(os.path.join(args.output_path, "RandomForest_best_params.txt"), 'w') as f:
            f.write(str(rf_best_params))

        results.append(["RandomForest", train_time, test_time, rf_val_acc, rf_test_acc])

        # XGBoost
        start = time.time()
        xgb, xgb_best_params, xgb_val_acc = tune_model(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
        }, X_train, y_train, X_val, y_val, "XGBoost")
        train_time = time.time() - start

        start = time.time()
        xgb_test_acc = evaluate_and_log(xgb, X_test, y_test, args.output_path, "XGBoost")
        test_time = time.time() - start

        with open(os.path.join(args.output_path, "XGBoost_best_params.txt"), 'w') as f:
            f.write(str(xgb_best_params))

        results.append(["XGBoost", train_time, test_time, xgb_val_acc, xgb_test_acc])

        # Neural Network
        start = time.time()
        nn, nn_best_params, nn_val_acc = tune_model(MLPClassifier(max_iter=500), {
            'hidden_layer_sizes': [(128,), (128, 64)],
            'alpha': [1e-4, 1e-3],
            'activation': ['relu', 'tanh']
        }, X_train, y_train, X_val, y_val, "NeuralNet")
        train_time = time.time() - start

        start = time.time()
        nn_test_acc = evaluate_and_log(nn, X_test, y_test, args.output_path, "NeuralNet")
        test_time = time.time() - start

        with open(os.path.join(args.output_path, "NeuralNet_best_params.txt"), 'w') as f:
            f.write(str(nn_best_params))

        results.append(["NeuralNet", train_time, test_time, nn_val_acc, nn_test_acc])

    except Exception as e:
        logging.exception("Error during training and evaluation")

    # Save summary CSV
    df = pd.DataFrame(results, columns=["Model", "Train Time (s)", "Test Time (s)", "Val Accuracy", "Test Accuracy"])
    df.to_csv(os.path.join(args.output_path, "summary_results.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to directory containing .pt embeddings')
    parser.add_argument('--splits_path', type=str, required=True, help='Path to directory containing train.txt/val.txt/test.txt')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')
    args = parser.parse_args()
    main(args)
