# Week 11 Hands-On: Simple Neural Network (Keras + MNIST)

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1. Load MNIST dataset (28x28 grayscale digit images, labels 0–9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# 2. Normalize input data (0–255 -> 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. One-hot encode the labels (e.g., 3 -> [0,0,0,1,0,0,0,0,0,0])
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. Build a simple Neural Network
model = Sequential([
    Flatten(input_shape=(28, 28)),         # Flatten 28x28 image to 784 vector
    Dense(128, activation='relu'),         # Hidden Layer with 128 neurons
    Dense(64, activation='relu'),          # Another hidden layer
    Dense(10, activation='softmax')        # Output Layer for 10 classes
])

# 5. Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=32
)

# 7. Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# 8. Predict some test samples
import numpy as np
import matplotlib.pyplot as plt

predictions = model.predict(x_test[:5])

for i in range(5):
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.axis("off")
    plt.show()
