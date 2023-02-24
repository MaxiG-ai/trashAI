# this script converts the tensorflow model to a tensorflow lite model
import tensorflow as tf

# Path to the saved model
saved_model_dir = 'models/model_2023-02-11_15-06-27'

#convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model
with open(saved_model_dir + '.tflite', 'wb') as f:
  f.write(tflite_model)