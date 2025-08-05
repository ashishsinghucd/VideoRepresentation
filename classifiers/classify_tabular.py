# evaluation_classifiers.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import argparse


def load_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    video_ids, labels = zip(*[line.strip().split() for line in lines])
    return list(video_ids), list(labels)


def load_embeddings(video_ids, label_list, emb_folder):
    X, y = [], []
    for vid, label in zip(video_ids, label_list):
        emb_path = os.path.join(emb_folder, vid + ".pt")
        if os.path.exists(emb_path):
            emb = torch.load(emb_path).numpy()
            X.append(emb)
            y.append(label)
    return np.stack(X), np.array(y)


def evaluate_classifiers(X_train, y_train, X_val, y_val, class_names, output_csv="results.csv"):
    classifiers = {
        'RandomForest': (RandomForestClassifier(), {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }),
        'SVM': (SVC(probability=True), {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10]
        }),
        'XGBoost': (XGBClassifier(objective='multi:softmax', eval_metric='mlogloss'), {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.05, 0.1]
        })
    }

    results = []

    for name, (clf, param_grid) in classifiers.items():
        print(f"Tuning {name}...")
        grid = GridSearchCV(clf, param_grid, cv=3, scoring='f1_weighted', verbose=1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        preds = best_model.predict(X_val)

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average=None)
        precision = precision_score(y_val, preds, average=None)
        recall = recall_score(y_val, preds, average=None)

        print(f"Best Parameters for {name}: {grid.best_params_}")
        print(f"Results for {name}:")
        print(classification_report(y_val, preds, target_names=class_names))

        for i, cls in enumerate(class_names):
            results.append({
                'Model': name,
                'Class': cls,
                'Accuracy': acc,
                'F1': f1[i],
                'Precision': precision[i],
                'Recall': recall[i]
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_nn(X_train, y_train, X_val, y_val, class_names, epochs=20, lr=0.001, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(X_train.shape[1], len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                out = model(batch_X)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())

        print(f"Epoch {epoch+1}/{epochs}:")
        print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=str, required=True, help="Path to folder with train.txt, test.txt, val.txt")
    parser.add_argument("--embed_dir", type=str, required=True, help="Path to folder containing train_embeddings, test_embeddings, val_embeddings")
    args = parser.parse_args()

    le = LabelEncoder()

    splits = {}
    for split in ["train", "test", "val"]:
        txt_path = os.path.join(args.split_dir, f"{split}.txt")
        emb_path = os.path.join(args.embed_dir, f"{split}_embeddings")
        video_ids, labels = load_labels(txt_path)
        if split == "train":
            le.fit(labels)
        labels_encoded = le.transform(labels)
        X, y = load_embeddings(video_ids, labels, emb_path)
        splits[split] = (X, labels_encoded)

    class_names = list(le.classes_)

    # Classical classifiers with hyperparameter tuning
    evaluate_classifiers(*splits["train"], splits["val"][0], splits["val"][1], class_names)

    # Train PyTorch MLP
    # train_nn(*splits["train"], *splits["val"], class_names)
