import tensorflow as tf

# Load the .h5 model
model = tf.keras.models.load_model('model.h5')

# Save the model in SavedModel format
model.save('saved_model', save_format='tf')

print("Model successfully converted to SavedModel format!")