# ANN - Artificial Neural Network
# Example: Iris Flower Classification

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Shape:", X.shape)
print("Classes:", iris.target_names)


# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# 4. Build ANN Model
# --------------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(4,)),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        8,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )
])


# --------------------------------------------------
# 5. Compile Model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 6. Train Model
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.20,
    verbose=1
)


# --------------------------------------------------
# 7. Evaluate Model
# --------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")


# --------------------------------------------------
# 8. Predictions
# --------------------------------------------------

y_probability = model.predict(X_test)

y_pred = np.argmax(
    y_probability,
    axis=1
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# --------------------------------------------------
# 9. Predict One New Flower
# --------------------------------------------------

sample = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

predicted_class = np.argmax(prediction)

print(
    "\nPredicted Flower:",
    iris.target_names[predicted_class]
)