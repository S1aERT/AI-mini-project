import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import uvicorn
from fastapi import FastAPI

app = FastAPI()

# Define file paths
MODEL_PATH = "saved_model"  # Optimized TensorFlow SavedModel format
DATASET_PATH = "dataset.csv"

# TensorFlow optimizations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable TensorFlow logs
tf.config.optimizer.set_jit(True)  # Enable XLA

# Load dataset
data = pd.read_csv(DATASET_PATH)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['TranslatedInstructions'])
max_sequence_length = 200

# Lazy model loading
model = None
def get_model():
    global model
    if model is None:
        print("Loading optimized model, please wait...")
        start_time = time.time()
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully in {time.time() - start_time:.2f} sec!")
    return model

# Function to generate a recipe
def generate_recipe(seed_text: str, next_words: int = 50):
    start_time = time.time()
    model = get_model()
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        
        # Optimize model inference
        predict_start = time.time()
        predicted = model.predict(token_list, verbose=0)
        predict_end = time.time()

        predicted_word = np.argmax(predicted, axis=-1)[0]
        output_word = next((word for word, index in tokenizer.word_index.items() if index == predicted_word), "")
        
        seed_text += " " + output_word

    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.2f} sec")
    print(f"Model inference time per step: {predict_end - predict_start:.4f} sec")
    
    return seed_text

# API Endpoints
@app.get("/")
def home():
    return {"message": "AI Recipe Generator is Running!"}

@app.post("/generate_recipe")
def generate(seed_text: str, next_words: int = 50):
    generated_recipe = generate_recipe(seed_text, next_words)
    return {"generated_recipe": generated_recipe}

# Run only when executed as a script
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, workers=4)
