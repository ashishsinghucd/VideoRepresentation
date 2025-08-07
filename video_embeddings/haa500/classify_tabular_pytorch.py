import os
import time
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Setup Logging ---
# Configure logging to show timestamp, level, and message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Also print to console
    ]
)


# --- 2. Custom PyTorch Dataset ---
class VideoEmbeddingDataset(Dataset):
    """
    Custom PyTorch Dataset for loading video embeddings and labels.
    """

    def __init__(self, split_file, embeddings_dir, label_map=None, transform=None):
        """
        Args:
            split_file (string): Path to the train.txt, val.txt, or test.txt file.
            embeddings_dir (string): Directory with all the .npz embedding files.
            label_map (dict, optional): A dictionary mapping class names to integers.
                                        If None, it will be created from the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.embeddings_dir = embeddings_dir
        self.transform = transform
        self.samples = []
        self.labels = []

        # Read the split file and load sample paths and labels
        try:
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    filename = line
                    # --- MODIFIED: Extract classname from filename ---
                    # e.g., 'jumping_jack_001' -> 'jumping_jack'
                    try:
                        classname = filename.rsplit('_', 1)[0]
                    except IndexError:
                        logging.warning(
                            f"Could not extract classname from filename: {filename} in {split_file}. Skipping.")
                        continue

                    embedding_path = os.path.join(self.embeddings_dir, f"{filename}.npz")

                    if os.path.exists(embedding_path):
                        self.samples.append(embedding_path)
                        self.labels.append(classname)
                    else:
                        logging.warning(f"Embedding file not found: {embedding_path}")

            if not self.samples:
                raise FileNotFoundError(
                    f"No valid samples were found for split file: {split_file}. Check paths and embedding directory.")

        except FileNotFoundError:
            logging.error(f"Split file not found at {split_file}")
            raise

        # Create or use the label map
        if label_map is None:
            self.classes = sorted(list(set(self.labels)))
            self.label_map = {label: i for i, label in enumerate(self.classes)}
        else:
            self.label_map = label_map
            self.classes = sorted(list(self.label_map.keys()))

        # Convert string labels to integer indices
        self.encoded_labels = [self.label_map[label] for label in self.labels]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        embedding_path = self.samples[idx]
        # Load the embedding from the .npz file
        # Assumes the data is stored with the key 'embedding'
        embedding = np.load(embedding_path)['data']
        label = self.encoded_labels[idx]

        # Apply the transformation (e.g., standardization)
        if self.transform:
            embedding = self.transform.transform(embedding.reshape(1, -1)).flatten()

        return torch.FloatTensor(embedding), torch.LongTensor([label]).squeeze()


# --- 3. Neural Network Model ---
class ClassifierNet(nn.Module):
    """
    A flexible multi-layer neural network for classification.
    """

    def __init__(self, input_dim, num_classes, hidden_layers=[512, 256], dropout_rate=0.5):
        """
        Args:
            input_dim (int): Dimension of the input features.
            num_classes (int): Number of output classes.
            hidden_layers (list of int): List of sizes for each hidden layer.
            dropout_rate (float): Dropout probability.
        """
        super(ClassifierNet, self).__init__()
        layers = []

        # Dynamically create layers based on the hidden_layers list
        last_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Batch normalization for stability
            layers.append(nn.ReLU(inplace=True))  # Non-linearity
            layers.append(nn.Dropout(dropout_rate))  # Regularization
            last_dim = hidden_dim

        # Add the final output layer
        layers.append(nn.Linear(last_dim, num_classes))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --- 4. Training and Evaluation Functions ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()


def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the given dataset."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to calculate gradients during evaluation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item(), all_preds, all_labels


# --- 5. Plotting Function ---
def plot_curves(history, output_folder):
    """Plots and saves training/validation loss and accuracy curves."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plotting training and validation loss
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plotting training and validation accuracy
    ax2.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plot_path = os.path.join(output_folder, 'training_curves.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Training curves saved to {plot_path}")


# --- 6. Main Execution Block ---
def main(args):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_folder, exist_ok=True)

        # --- Device Configuration ---
        use_cuda = torch.cuda.is_available() and args.use_gpu
        device = torch.device("cuda" if use_cuda else "cpu")
        logging.info(f"Using device: {device}")

        # --- Data Loading and Preparation ---
        train_split_file = os.path.join(args.split_folder, 'train.txt')
        val_split_file = os.path.join(args.split_folder, 'val.txt')
        test_split_file = os.path.join(args.split_folder, 'test.txt')

        # Create training dataset to establish label map and scaler
        train_dataset = VideoEmbeddingDataset(train_split_file, args.embeddings_folder)
        label_map = train_dataset.label_map
        class_names = train_dataset.classes
        num_classes = len(class_names)

        # Get all training data to fit the scaler
        X_train_list = [np.load(p)["data"] for p in train_dataset.samples]
        X_train_raw = np.array(X_train_list)

        # Get input dimension from the first sample
        input_dim = X_train_raw.shape[1]

        # --- Data Standardization ---
        logging.info("Standardizing data...")
        scaler = StandardScaler()
        scaler.fit(X_train_raw)  # Fit ONLY on training data

        # Apply the fitted scaler to the training dataset instance
        train_dataset.transform = scaler

        # Create validation and test datasets using the same label map and scaler
        val_dataset = VideoEmbeddingDataset(val_split_file, args.embeddings_folder, label_map=label_map,
                                            transform=scaler)
        test_dataset = VideoEmbeddingDataset(test_split_file, args.embeddings_folder, label_map=label_map,
                                             transform=scaler)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=True)

        logging.info(f"Training data shape (samples, features): ({len(train_dataset)}, {input_dim})")
        logging.info(f"Validation data shape (samples, features): ({len(val_dataset)}, {input_dim})")
        logging.info(f"Test data shape (samples, features): ({len(test_dataset)}, {input_dim})")
        logging.info(f"Number of classes: {num_classes}")

        # --- Model, Loss, and Optimizer ---
        model = ClassifierNet(input_dim, num_classes, [128]).to(device)
        logging.info(f"Model architecture:\n{model}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adam with L2 regularization

        # --- Training and Validation Loop ---
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0

        training_start_time = time.time()

        for epoch in range(args.epochs):
            logging.info(f"--- Epoch {epoch + 1}/{args.epochs} ---")

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Evaluate on the validation set
            val_start_time = time.time()
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            val_time = time.time() - val_start_time
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} (Evaluation took {val_time:.2f}s)")

            # Check for best results on validation set and save model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logging.info(f"New best validation accuracy: {best_val_acc:.4f}. Saving model...")
                torch.save(model.state_dict(), os.path.join(args.output_folder, 'best_model.pth'))

        training_time = time.time() - training_start_time
        logging.info(f"\nTotal training time: {training_time:.2f} seconds")
        logging.info(f"Best validation accuracy achieved: {best_val_acc:.4f}")

        # --- Final Evaluation on Test Set ---
        logging.info("\n--- Evaluating best model on the TEST set ---")
        # Load the best model saved during training
        model.load_state_dict(torch.load(os.path.join(args.output_folder, 'best_model.pth')))

        test_start_time = time.time()
        test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
        test_time = time.time() - test_start_time

        logging.info(f"Final Test Loss: {test_loss:.4f}")
        logging.info(f"Final Test Accuracy: {test_acc:.4f}")
        logging.info(f"Final Test evaluation took {test_time:.2f}s")

        # --- Save Final Test Metrics ---
        logging.info("\n--- Final Test Set Performance Metrics ---")
        report_str = classification_report(y_true, y_pred, target_names=class_names)
        logging.info("Classification Report:\n" + report_str)

        # Flatten the report dictionary for easier CSV export
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_df_data = []
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                row = {'class': class_name, **metrics}
                report_df_data.append(row)

        report_df = pd.DataFrame(report_df_data)
        csv_path = os.path.join(args.output_folder, 'final_test_metrics.csv')
        report_df.to_csv(csv_path, index=False)
        logging.info(f"Final test metrics saved to {csv_path}")

        # --- Save Plots ---
        plot_curves(history, args.output_folder)

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Neural Network on VideoMAE Embeddings")
    parser.add_argument('--split_folder', type=str, required=True,
                        help="Folder containing train.txt, val.txt, and test.txt")
    parser.add_argument('--embeddings_folder', type=str, required=True,
                        help="Folder containing the .npz embedding files")
    parser.add_argument('--output_folder', type=str, required=True, help="Folder to save results (model, plots, csv)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and testing")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--no_gpu', action='store_false', dest='use_gpu',
                        help="Flag to disable GPU usage even if available")

    # Example usage from command line:
    # python your_script_name.py --split_folder ./path/to/splits --embeddings_folder ./path/to/embeddings --output_folder ./results --epochs 100 --batch_size 32

    args = parser.parse_args()
    main(args)
