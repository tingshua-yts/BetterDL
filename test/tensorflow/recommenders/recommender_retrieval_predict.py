import tensorflow as tf
import os
path = os.path.join("tmp/", "retrieval_model")
# Load it back; can also be done in TensorFlow Serving.
loaded = tf.saved_model.load(path)

# Pass a user id in, get top predicted movie titles back.
scores, titles = loaded(["42"])

print(f"Recommendations: {titles[0][:3]}")
