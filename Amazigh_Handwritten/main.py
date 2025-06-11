import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def relu(x):
    """ReLU activation: élément par élément max(0, x)."""
    assert isinstance(x, np.ndarray), "Input must be numpy array"
    result = np.maximum(0, x)
    assert np.all(result >= 0)
    return result

def relu_derivative(x):
    """Dérivée de ReLU : 1 si x > 0, 0 sinon."""
    assert isinstance(x, np.ndarray)
    result = (x > 0).astype(float)
    assert np.all((result == 0) | (result == 1))
    return result

def softmax(x):
    """Softmax activation sur les lignes."""
    assert isinstance(x, np.ndarray)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    assert np.all(result >= 0) and np.all(result <= 1)
    assert np.allclose(np.sum(result, axis=1), 1)
    return result


class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        layer_sizes : [input_dim, hidden1, ..., output_dim]
        """
        assert isinstance(layer_sizes, list) and len(layer_sizes) >= 2
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        self.weights = []
        self.biases = []
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """Propagation vers l'avant, stocke Z et A."""
        A = X
        self.activations = [A]
        self.z_values = []

        for i in range(len(self.weights) - 1):
            Z = A.dot(self.weights[i]) + self.biases[i]
            A = relu(Z)
            self.z_values.append(Z)
            self.activations.append(A)

        Z = A.dot(self.weights[-1]) + self.biases[-1]
        A = softmax(Z)
        self.z_values.append(Z)
        self.activations.append(A)
        return A

    def compute_loss(self, y_true, y_pred):
        """Cross-entropy catégorielle moyenne."""
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def compute_accuracy(self, y_true, y_pred):
        """Exactitude = proportion correctes."""
        preds = np.argmax(y_pred, axis=1)
        truths = np.argmax(y_true, axis=1)
        return np.mean(preds == truths)

    def backward(self, X, y, y_pred):
        """Rétropropagation : calcul des gradients."""
        m = X.shape[0]
        L = len(self.weights)
        self.d_weights = [None] * L
        self.d_biases = [None] * L


        dZ = y_pred - y
        self.d_weights[-1] = self.activations[-2].T.dot(dZ) / m
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m


        for i in reversed(range(L-1)):
            dA = dZ.dot(self.weights[i+1].T)
            dZ = dA * relu_derivative(self.z_values[i])
            self.d_weights[i] = self.activations[i].T.dot(dZ) / m
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m


        for i in range(L):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        """Entraînement via mini-batch SGD."""
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(epochs):
            # mélange du dataset
            perm = np.random.permutation(X.shape[0])
            X_sh, y_sh = X[perm], y[perm]

            loss_sum = 0
            for i in range(0, X.shape[0], batch_size):
                xb = X_sh[i:i+batch_size]
                yb = y_sh[i:i+batch_size]
                pred = self.forward(xb)
                loss_sum += self.compute_loss(yb, pred)
                self.backward(xb, yb, pred)

            train_pred = self.forward(X)
            val_pred = self.forward(X_val)

            train_loss = loss_sum / (X.shape[0]//batch_size)
            train_acc = self.compute_accuracy(y, train_pred)
            val_loss = self.compute_loss(y_val, val_pred)
            val_acc = self.compute_accuracy(y_val, val_pred)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train loss {train_loss:.4f}, acc {train_acc:.4f} | Val loss {val_loss:.4f}, acc {val_acc:.4f}")
        return train_losses, val_losses, train_accs, val_accs

    def predict(self, X):
        """Retourne la classe (idx) prédite pour chaque échantillon."""
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)


data_dir = r'C:\Users\nnaji\Desktop\AMHCD - Copy\AMHCD\images'
csv_path = os.path.join(data_dir, 'labels-map.csv')

if os.path.exists(csv_path):
    labels_df = pd.read_csv(csv_path)
else:

    image_paths, labels = [], []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for fname in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, fname))
                labels.append(label)
    labels_df = pd.DataFrame({'image_path': image_paths, 'label': labels})

print(f"Loaded {len(labels_df)} samples, {labels_df['label'].nunique()} classes.")


le = LabelEncoder()
labels_df['label_encoded'] = le.fit_transform(labels_df['label'])
num_classes = len(le.classes_)

def load_and_preprocess_image(path, size=(32,32)):
    """Charge, convertit en gris, redimensionne et normalise."""
    assert os.path.exists(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0

X = np.array([load_and_preprocess_image(p) for p in labels_df['image_path']])
y = labels_df['label_encoded'].values
X = X.reshape(X.shape[0], -1)  # aplatir 32x32 -> 1024 vecteurs

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

ohe = OneHotEncoder(sparse_output=False)
y_train_ohe = ohe.fit_transform(y_train.reshape(-1,1))
y_val_ohe = ohe.transform(y_val.reshape(-1,1))
y_test_ohe = ohe.transform(y_test.reshape(-1,1))

print(f"Train/Val/Test sizes: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]} samples")

layer_sizes = [X_train.shape[1], 64, 32, num_classes]
nn = MultiClassNeuralNetwork(layer_sizes, learning_rate=0.01)
train_losses, val_losses, train_accs, val_accs = nn.train(X_train, y_train_ohe, X_val, y_val_ohe, epochs=100, batch_size=32)

y_pred_idx = nn.predict(X_test)
print("\nClassification report (test set):")
print(classification_report(y_test, y_pred_idx, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred_idx)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion (Test set)")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses, label="Val Loss")
ax1.set_title("Perte")
ax1.set_xlabel("Époque")
ax1.set_ylabel("Loss")
ax1.legend()
ax2.plot(train_accs, label="Train Acc")
ax2.plot(val_accs, label="Val Acc")
ax2.set_title("Exactitude")
ax2.set_xlabel("Époque")
ax2.set_ylabel("Accuracy")
ax2.legend()
plt.tight_layout()
plt.show()
