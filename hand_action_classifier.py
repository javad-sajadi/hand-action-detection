"""
hand_action_classifier.py - Convolutional Neural Network for Hand Gesture and Sign Language Detection
Author: Javad Sajadi
Repository: javad-sajadi/hand-action-detection

Trains, evaluates, and tests a deep CNN model for recognizing hand gesture digits
from the Sign Language Digits benchmark dataset (64x64 grayscale).
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization


def load_dataset(data_dir: str = "."):
    """
    Loads X.npy and Y.npy from the specified directory.
    Falls back to generating synthetic demonstration samples if dataset is absent.
    """
    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "Y.npy")

    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"Loading dataset from: {data_dir}")
        X = np.load(x_path)
        Y = np.load(y_path)
    else:
        print(f"Dataset files not found in '{data_dir}'. Generating synthetic 64x64 demo dataset for verification...")
        np.random.seed(42)
        n_samples = 200
        X = np.random.uniform(0.0, 1.0, size=(n_samples, 64, 64)).astype(np.float32)
        labels = np.random.randint(0, 10, size=n_samples)
        Y = np.eye(10)[labels]

    # Normalize if values are in [0, 255]
    if np.max(X) > 1.0:
        X = X / 255.0

    # Ensure shape is (N, 64, 64, 1)
    if X.ndim == 3:
        X = X.reshape(-1, 64, 64, 1)

    return X, Y


def build_cnn_model(input_shape=(64, 64, 1), num_classes=10) -> Sequential:
    """Builds an optimized deep CNN architecture with Max-Pooling and Dropout."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train(data_dir: str = ".", epochs: int = 20, batch_size: int = 32, model_save_path: str = "hand_gesture_cnn.h5"):
    """Trains the CNN model and evaluates on a holdout test split."""
    X, Y = load_dataset(data_dir)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
    print(f"Dataset split: {len(x_train)} training samples, {len(x_test)} test samples.")

    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=Y.shape[1])
    model.summary()

    print(f"\nTraining CNN for {epochs} epochs (batch size {batch_size})...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

    model.save(model_save_path)
    print(f"Trained model saved to: {model_save_path}")
    return model


def predict(model_path: str = "hand_gesture_cnn.h5", data_dir: str = ".", sample_idx: int = 0):
    """Predicts hand gesture class for a given sample."""
    if not os.path.exists(model_path):
        print(f"Model '{model_path}' not found. Training model first...")
        train(data_dir=data_dir, epochs=5, model_save_path=model_path)

    model = tf.keras.models.load_model(model_path)
    X, Y = load_dataset(data_dir)

    sample = X[sample_idx:sample_idx + 1]
    prob = model.predict(sample)[0]
    pred_class = int(np.argmax(prob))
    true_class = int(np.argmax(Y[sample_idx]))

    print(f"\nSample #{sample_idx}:")
    print(f"Predicted Class: {pred_class} (Confidence: {prob[pred_class]*100:.2f}%)")
    print(f"Ground Truth:    {true_class}")
    return pred_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Action & Sign Language Digit Classifier")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing X.npy and Y.npy")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--train", action="store_true", help="Train the CNN model")
    parser.add_argument("--predict", type=int, default=None, help="Sample index to predict")
    parser.add_argument("--model-path", type=str, default="hand_gesture_cnn.h5", help="Saved model file")
    args = parser.parse_args()

    if args.predict is not None:
        predict(model_path=args.model_path, data_dir=args.data_dir, sample_idx=args.predict)
    else:
        train(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, model_save_path=args.model_path)
