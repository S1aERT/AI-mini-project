import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
# Load dataset
data = pd.read_csv(r"C:\Users\sarth\Downloads\AI-Recipe-Generator\dataset.csv")

# Check dataset structure
print(data.head())

# Ensure the dataset contains 'TranslatedInstructions' column
if 'TranslatedInstructions' not in data.columns:
    raise ValueError("Dataset must have a 'TranslatedInstructions' column.")

# Drop duplicate recipes
data.drop_duplicates(subset=['TranslatedInstructions'], inplace=True)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['TranslatedInstructions'])

# Save tokenizer for future use
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Convert text into sequences
sequences = tokenizer.texts_to_sequences(data['TranslatedInstructions'])

# Define max sequence length
max_sequence_length = 200
sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
# Define model architecture
vocab_size = len(tokenizer.word_index) + 1

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 100, input_length=max_sequence_length-1),
    tf.keras.layers.LSTM(150, return_sequences=True),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display model summary
model.summary()
# Prepare input and output data
X = sequences[:, :-1]
y = sequences[:, -1]

# Train model
model.fit(X, y, epochs=10, batch_size=64)
model.save(r'C:\Users\sarth\Downloads\AI-Recipe-Generator/recipe_generation_model.h5')