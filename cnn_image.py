# RNN - Recurrent Neural Network
# Example: IMDB Sentiment Classification

import tensorflow as tf


# --------------------------------------------------
# 1. Load IMDB Dataset
# --------------------------------------------------

vocab_size = 10000
max_length = 200

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.imdb.load_data(
        num_words=vocab_size
    )
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# --------------------------------------------------
# 2. Pad Sequences
# --------------------------------------------------

X_train = tf.keras.utils.pad_sequences(
    X_train,
    maxlen=max_length,
    padding="post"
)

X_test = tf.keras.utils.pad_sequences(
    X_test,
    maxlen=max_length,
    padding="post"
)


# --------------------------------------------------
# 3. Build RNN Model
# --------------------------------------------------

model = tf.keras.Sequential([

    # Convert word IDs into vectors
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=32,
        input_length=max_length
    ),

    # Remember information from previous words
    tf.keras.layers.SimpleRNN(
        32
    ),

    # Binary classification
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# --------------------------------------------------
# 4. Compile Model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 5. Train Model
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.20
)


# --------------------------------------------------
# 6. Evaluate Model
# --------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    "\nRNN Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# --------------------------------------------------
# 7. Prediction
# --------------------------------------------------

prediction = model.predict(
    X_test[:1]
)

sentiment = (
    "Positive"
    if prediction[0][0] >= 0.5
    else "Negative"
)

print(
    "\nPredicted Sentiment:",
    sentiment
)

print(
    "Prediction Score:",
    round(float(prediction[0][0]), 4)
)