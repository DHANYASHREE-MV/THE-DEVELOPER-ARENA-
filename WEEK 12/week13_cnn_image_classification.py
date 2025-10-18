# Week 13 - CNN for Image Classification
# Author: baby

# ========== IMPORT LIBRARIES ==========
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# ========== STEP 1: LOAD & PREPARE DATA ==========
print("📥 Loading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values (0–255 → 0–1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# CIFAR-10 class names
class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"✅ Data loaded: {X_train.shape[0]} training images, {X_test.shape[0]} testing images.")

# ========== STEP 2: VISUALIZE SOME IMAGES ==========
plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_train[i])
    plt.title(class_names[int(y_train[i])])
    plt.axis("off")
plt.suptitle("🖼️ Sample CIFAR-10 Images")
plt.show()

# ========== STEP 3: BUILD CNN MODEL ==========
model = models.Sequential([
    # Convolution + Pooling Layer 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # Convolution + Pooling Layer 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Convolution Layer 3
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten and Fully Connected Layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # 10 output classes
])

# ========== STEP 4: COMPILE MODEL ==========
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# ========== STEP 5: TRAIN MODEL ==========
print("\n🚀 Training started...")
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_test, y_test),
                    batch_size=64)
print("✅ Training complete.")

# ========== STEP 6: EVALUATE MODEL ==========
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")

# ========== STEP 7: VISUALIZE TRAINING RESULTS ==========
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend(), plt.title('📈 Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(), plt.title('📉 Loss over Epochs')
plt.show()

# ========== STEP 8: PREDICT SOME IMAGES ==========
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(X_test[:8])

plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_test[i])
    plt.title(f"Pred: {class_names[np.argmax(predictions[i])]}\nTrue: {class_names[int(y_test[i])]}")
    plt.axis("off")
plt.suptitle("🔮 CNN Predictions on Test Images")
plt.show()

# ========== STEP 9: SAVE MODEL ==========
model.save("cnn_cifar10_model.h5")
print("💾 Model saved as cnn_cifar10_model.h5")

print("\n✨ DONE — CNN image classification completed successfully!")
