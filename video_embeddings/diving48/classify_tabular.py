import os
import time
import glob
import torch
import numpy as np
import pandas as pd
import logging
import argparse
import pickle
import json
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def load_json_split(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [(item['vid_name'], item['label']) for item in data]

def load_data_from_json(split_data, embedding_folder):
    X, y = [], []
    for vid_name, label in split_data:
        file_path = os.path.join(embedding_folder, vid_name + '.npz')
        if not os.path.exists(file_path):
            logging.warning(f"Missing file: {file_path}")
            continue
        try:
            vec = np.load(file_path)["data"]
            X.append(vec)
            y.append(label)
        except Exception as e:
            logging.warning(f"Failed to load {file_path}: {e}")
    return np.array(X), np.array(y)

def tune_model(model, param_grid, X_train, y_train, model_name):
    logging.info(f"Tuning {model_name}...")
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    logging.info(f"Best {model_name} params: {grid.best_params_}")
    return best_model, grid.best_params_

def evaluate_and_log(model, X_test, y_test, output_path, model_name):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_path, f"{model_name}_report.csv"))
    logging.info(f"Saved evaluation metrics for {model_name}.")
    return accuracy_score(y_test, y_pred)

def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    model_dir = os.path.join(args.output_path, "model_weights")
    os.makedirs(model_dir, exist_ok=True)

    train_data = load_json_split(os.path.join(args.splits_path, 'Diving48_V2_train.json'))
    test_data = load_json_split(os.path.join(args.splits_path, 'Diving48_V2_test.json'))

    X_train, y_train = load_data_from_json(train_data, args.input_path)
    X_test, y_test = load_data_from_json(test_data, args.input_path)

    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))

    results = []

    try:
        classifiers = {
            # "RandomForest": (RandomForestClassifier(), {
            #     'n_estimators': [100, 200],
            #     'max_depth': [None, 10, 20]
            # }),
            "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), {
                'n_estimators': [100, 200],
                'max_depth': [3, 6]
            }),
            "NeuralNet": (MLPClassifier(max_iter=500), {
                'hidden_layer_sizes': [(128,), (128, 64)],
                'alpha': [1e-3],
                'activation': ['tanh']
            })
        }

        for model_name, (clf, params) in classifiers.items():
            logging.info(f"\n==== Processing {model_name} ====")
            start = time.time()
            best_model, best_params = tune_model(clf, params, X_train, y_train, model_name)
            train_time = time.time() - start

            start = time.time()
            test_acc = evaluate_and_log(best_model, X_test, y_test, args.output_path, model_name)
            test_time = time.time() - start

            with open(os.path.join(args.output_path, f"{model_name}_best_params.txt"), 'w') as f:
                f.write(str(best_params))

            with open(os.path.join(model_dir, f"{model_name}_best_model.pkl"), 'wb') as f:
                pickle.dump(best_model, f)

            results.append([model_name, train_time, test_time, test_acc])

    except Exception as e:
        logging.exception("Error during training and evaluation")

    df = pd.DataFrame(results, columns=["Model", "Train Time (s)", "Test Time (s)", "Test Accuracy"])
    df.to_csv(os.path.join(args.output_path, "summary_results.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to directory containing .pt embeddings')
    parser.add_argument('--splits_path', type=str, required=True, help='Path to directory containing Diving48_V2_train.json/test.json')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')
    args = parser.parse_args()
    main(args)
