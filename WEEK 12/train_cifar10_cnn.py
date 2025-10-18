"""
train_cifar10_cnn.py
A full example: build, train, evaluate, and save a CNN on CIFAR-10.

Usage:
    python train_cifar10_cnn.py

Notes:
 - Uses TensorFlow / Keras.
 - Defaults: 50 epochs, batch_size=128 (change below).
 - If running on CPU and it's slow, reduce epochs or use Colab/GPU.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers

# ---------------------------
# Hyperparameters / Config
# ---------------------------
BATCH_SIZE = 128
EPOCHS = 50
IMG_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------------------------
# Load CIFAR-10
# ---------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()
print(f"Train shape: {x_train.shape}, Train labels: {y_train.shape}")
print(f"Test shape:  {x_test.shape}, Test labels:  {y_test.shape}")

# ---------------------------
# Preprocessing + Data Augmentation
# ---------------------------
# Normalize images to [0,1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Simple data augmentation pipeline (on-the-fly)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.06, 0.06),
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.05, 0.05),
], name="data_augmentation")

# ---------------------------
# Model: Conv blocks with BatchNorm + Dropout
# ---------------------------
def build_cnn(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    weight_decay = 1e-4
    inputs = layers.Input(shape=input_shape)

    x = data_augmentation(inputs)            # data augmentation only during training
    # Block 1
    x = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.35)(x)

    # Block 3
    x = layers.Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.45)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")
    return model

model = build_cnn()
model.summary()

# ---------------------------
# Compile model
# ---------------------------
# Use sparse categorical crossentropy (labels are integers not one-hot)
optimizer = optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------
# Callbacks: early stop, reduce lr, model checkpoint
# ---------------------------
checkpoint_path = os.path.join(MODEL_DIR, "best_cifar10_cnn.h5")
cb_checkpoint = callbacks.ModelCheckpoint(checkpoint_path,
                                          monitor='val_accuracy',
                                          save_best_only=True,
                                          verbose=1)

cb_reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
cb_early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)

# ---------------------------
# Train
# ---------------------------
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[cb_checkpoint, cb_reduce_lr, cb_early_stop],
    shuffle=True,
    verbose=2
)

# ---------------------------
# Save final model
# ---------------------------
final_model_path = os.path.join(MODEL_DIR, "final_cifar10_cnn.h5")
model.save(final_model_path)
print("Saved final model to:", final_model_path)

# ---------------------------
# Evaluate on test set
# ---------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# ---------------------------
# Plot training curves
# ---------------------------
def plot_history(history_obj, save_path="training_curves.png"):
    h = history_obj.history
    epochs_ran = range(1, len(h['loss']) + 1)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_ran, h['loss'], label='train loss')
    plt.plot(epochs_ran, h['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_ran, h['accuracy'], label='train acc')
    plt.plot(epochs_ran, h['val_accuracy'], label='val acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

plot_history(history, save_path=os.path.join(MODEL_DIR, "training_curves.png"))

# ---------------------------
# Show some sample predictions
# ---------------------------
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def show_sample_predictions(model, X, y, n=12):
    idx = np.random.choice(len(X), size=n, replace=False)
    Xs = X[idx]
    ys = y[idx]
    preds = model.predict(Xs)
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(12,6))
    for i in range(n):
        plt.subplot(3, 4, i+1)
        plt.imshow(Xs[i])
        true = class_names[ys[i]]
        pred = class_names[pred_labels[i]]
        color = 'green' if pred == true else 'red'
        plt.title(f"T:{true}\nP:{pred}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_sample_predictions(model, x_test, y_test, n=12)

# ---------------------------
# Done
# ---------------------------
print("Training complete. Models and plots saved to:", MODEL_DIR)
