import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)


for x in movies.take(1).as_numpy_iterator():
  pprint.pprint(x)

print(">>>>>>>>>>>>>>>>>>>>>>>>")

# rating 为训练数据,  其中movie_title  为feature、user_id为lable
# {'movie_title': b"One Flew Over the Cuckoo's Nest (1975)", 'user_id': b'138'}
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})

# movies是为了构建movies model的vocab使用，抽象上来讲是为了构建retrieval的candidate
# b'You So Crazy (1994)'
movies = movies.map(lambda x: x["movie_title"])

for x in ratings.take(1).as_numpy_iterator():
      pprint.pprint(x)


for x in movies.take(1).as_numpy_iterator():
  pprint.pprint(x)

#### shuffle ####
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

### 构建词表 #####
movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

unique_movie_titles[:10]


### 定义模型 ####

class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model, task: tfrs.tasks.Retrieval):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False):
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])

    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back. 相当于是label
    positive_movie_embeddings = self.movie_model(features["movie_title"])

    # The task computes the loss and the metrics. 根据query和positive candidate计算loss，目标是最大化positive_movie_embeddins的score
    return self.task(user_embeddings, positive_movie_embeddings)

#### 构建模型 ####
embedding_dimension = 32

# user model
# embedding table size为：[len(unique_user_ids), 32]
# input data shape 为：[batch_size]
# output data shape为:[batch_size, 32]
user_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

# movie model，vocab layer + embedding layer
# vocab  table size  为unique_movie_titles
# vocab table input data shape 为：[batch_size]
# vocab  table output data shape为 ：[batch_size]
# embedding table size为：[len(unique_movie_titles), 32]
# embedding input data shape 为：[batch_size]
# embedding output data shape为:[batch_size, 32]
movie_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_movie_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
  candidates=movies.batch(128).map(movie_model)
)

task = tfrs.tasks.Retrieval(
  metrics=metrics
)
model = MovielensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

#### 训练 ####
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
model.fit(cached_train, epochs=3)
model.evaluate(cached_test, return_dict=True)

#### predict ####
# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
)

# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")

#### save model ####
# Export the query model.
path = os.path.join("tmp/", "retrieval_model")

# Save the index.
tf.saved_model.save(index, path)

# Load it back; can also be done in TensorFlow Serving.
#loaded = tf.saved_model.load(path)

  # # Pass a user id in, get top predicted movie titles back.
  # scores, titles = loaded(["42"])

  # print(f"Recommendations: {titles[0][:3]}")
