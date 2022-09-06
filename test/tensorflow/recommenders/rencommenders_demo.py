from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import logging
import  logging
import datetime
#logging.basicConfig(format='[%(asctime)s] %(filename)s %(funcName)s():%(lineno)i [%(levelname)s] %(message)s', level=logging.DEBUG)


# 1) read data
# Ratings data.
ratings = tfds.load('movielens/100k-ratings', split="train", try_gcs= False, download=True, data_dir="tmp/data/movielens-100k")
# Features of all the available movies.
movies = tfds.load('movielens/100k-movies', split="train" , try_gcs= False, download=True, data_dir="tmp/data/movielens-100k")
"""
ratings type: <class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>, movies type: <class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>
"""
print(f"ratings type: {type(ratings)}, movies type: {type(movies)}")

# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"]
})
movies = movies.map(lambda x: x["movie_title"])



"""
ratings type: <class 'tensorflow.python.data.ops.dataset_ops.MapDataset'>,
movies type: <class 'tensorflow.python.data.ops.dataset_ops.MapDataset'>
ratings len: 100000
"""
print(f"ratings type: {type(ratings)}, movies type: {type(movies)}")
print(f"ratings len: {len(ratings)}")


user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))
movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
movie_titles_vocabulary.adapt(movies)

# 2) define model
class MovieLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      user_model: tf.keras.Model,
      movie_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.movie_model = movie_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) :
    # Define how the loss is computed.

    user_embeddings = self.user_model(features["user_id"])
    movie_embeddings = self.movie_model(features["movie_title"])

    return self.task(user_embeddings, movie_embeddings)

# Define user and movie models.
user_model = tf.keras.Sequential([
    user_ids_vocabulary,
    tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
])
movie_model = tf.keras.Sequential([
    movie_titles_vocabulary,
    tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
])

# Define your objectives.
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    movies.batch(128).map(movie_model)
  )
)

# 4) fit and eval

# Create a retrieval model.
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

logdir="tmp/logs/fit/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Train for 3 epochs.
model.fit(ratings.batch(4096), epochs=3,callbacks=[tensorboard_callback])

# Use brute-force search to set up retrieval using the trained representations.
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    movies.batch(100).map(lambda title: (title, model.movie_model(title))))

# Get some recommendations.
_, titles = index(np.array(["42"]))
print(f"Top 3 recommendations for user 42: {titles[0, :3]}")