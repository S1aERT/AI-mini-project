import tensorflow as tf

# Load the existing model
model = tf.keras.models.load_model("recipe_generation_model.h5")

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("recipe_generation_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully!")
