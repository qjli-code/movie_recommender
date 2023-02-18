# imports
import math
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

# code frameworks are adapted from Tensorflow recommenders tutorials
# see https://www.tensorflow.org/recommenders


class BucketsEmbedding(tf.keras.Model):
    """
    Class for discretized timestamp embedding
    """

    def __init__(self, timestamps, nbuckets=1000, embedding_size=32):
        super().__init__()
        self.nbuckets = nbuckets
        # timestamp discretization and embedding
        timestamp_buckets = self.calc_timestamp_buckets(timestamps)
        self.timestamp_embedding = tf.keras.Sequential([
            tf.keras.layers.Discretization(timestamp_buckets.tolist()),
            tf.keras.layers.Embedding(
                len(timestamp_buckets)+1, embedding_size)
        ])

    def calc_timestamp_buckets(self, timestamps):
        max_timestamp = timestamps.max()
        min_timestamp = timestamps.min()
        return np.linspace(min_timestamp, max_timestamp, num=self.nbuckets)

    def call(self, timestamps):
        return self.timestamp_embedding(timestamps)


class SinCosEncoder(tf.keras.Model):
    """
    Class for Sine-Cosine encoding of periodic data.
    """

    def __init__(self, max_val, output_size):
        super().__init__()
        self.max_val = max_val
        self.linear = tf.keras.layers.Dense(output_size)

    def sc_encoder(self, input):
        sin = tf.reshape(tf.math.sin(2*math.pi*input/self.max_val), (-1, 1))
        cos = tf.reshape(tf.math.cos(2*math.pi*input/self.max_val), (-1, 1))
        return tf.concat([sin, cos], axis=-1)

    def call(self, input):
        sin_cos = self.sc_encoder(input)
        return self.linear(sin_cos)


class UserModel(tf.keras.Model):
    """
    Class defining the user model which utilizes the following as features:
    1) user id embedding
    2) discretized timestamp embedding (buckets)
    3) standardized timestamp
    4) Since/Cosine encoding of month, weekday, day, and hours
    """

    def __init__(self, timestamps, emb_size=32,
                 time_encode_size=8, use_time=None,
                 layer_sizes=[64], debug=False, max_num_users=1_000):
        super().__init__()
        if debug:
            self.use_time = 'buckets'
        else:
            self.use_time = use_time
        # user id embedding
        # we use hashing with a large num_bins to avoid repeated values
        self.user_id_embedding = tf.keras.Sequential([
            tf.keras.layers.Hashing(num_bins=max_num_users),
            tf.keras.layers.Embedding(
                input_dim=max_num_users, output_dim=emb_size)
        ])
        # normalized timestamp
        self.standardized_timestamp = tf.keras.layers.Normalization(
            axis=None)
        self.standardized_timestamp.adapt(timestamps)
        # discretized timestamp embedding
        if self.use_time == 'buckets':
            self.timestamp_embedding = BucketsEmbedding(timestamps,
                                                        embedding_size=emb_size)
        # encoding month, weekday, day, hour
        elif self.use_time == 'sc_encoder':
            self.year_embedding = tf.keras.Sequential([
                tf.keras.layers.Hashing(num_bins=200),
                tf.keras.layers.Embedding(
                    input_dim=200, output_dim=time_encode_size)
            ])
            self.month_encoder = SinCosEncoder(12, time_encode_size)
            self.wkday_encoder = SinCosEncoder(7, time_encode_size)
            self.day_encoder = SinCosEncoder(31, time_encode_size)
            self.hour_encoder = SinCosEncoder(24, time_encode_size)
        elif self.use_time == 'both':
            self.timestamp_embedding = BucketsEmbedding(timestamps,
                                                        embedding_size=emb_size)
            self.year_embedding = tf.keras.Sequential([
                tf.keras.layers.Hashing(num_bins=200),
                tf.keras.layers.Embedding(
                    input_dim=200, output_dim=time_encode_size)
            ])
            self.month_encoder = SinCosEncoder(12, time_encode_size)
            self.wkday_encoder = SinCosEncoder(7, time_encode_size)
            self.day_encoder = SinCosEncoder(31, time_encode_size)
            self.hour_encoder = SinCosEncoder(24, time_encode_size)
        else:
            pass

        # add deep Dense layers to furhter transform embeddings
        self.dense_layers = tf.keras.Sequential()
        for ls in layer_sizes:
            self.dense_layers.add(tf.keras.layers.Dense(ls, activation='relu'))
        self.dense_layers.add(tf.keras.layers.Dense(emb_size))

    def call(self, inputs):
        if self.use_time == 'buckets':
            user_embedding = tf.concat([
                self.user_id_embedding(inputs['userId']),
                self.timestamp_embedding(inputs['timestamp']),
                tf.reshape(self.standardized_timestamp(
                    inputs['timestamp']), (-1, 1))
            ], axis=1)
        elif self.use_time == 'sc_encoder':
            user_embedding = tf.concat([
                self.user_id_embedding(inputs['userId']),
                self.year_embedding(inputs['year']),
                self.month_encoder(inputs['month']),
                self.wkday_encoder(inputs['weekday']),
                self.day_encoder(inputs['day']),
                self.hour_encoder(inputs['hour']),
                tf.reshape(self.standardized_timestamp(
                    inputs['timestamp']), (-1, 1))
            ], axis=1)
        elif self.use_time == 'both':
            user_embedding = tf.concat([
                self.user_id_embedding(inputs['userId']),
                self.timestamp_embedding(inputs['timestamp']),
                self.year_embedding(inputs['year']),
                self.month_encoder(inputs['month']),
                self.wkday_encoder(inputs['weekday']),
                self.day_encoder(inputs['day']),
                self.hour_encoder(inputs['hour']),
                tf.reshape(self.standardized_timestamp(
                    inputs['timestamp']), (-1, 1))
            ], axis=1)
        else:
            user_embedding = self.user_id_embedding(inputs['userId'])
        return self.dense_layers(user_embedding)


class MovieModel(tf.keras.Model):
    """
    Class defining the movie model which combines the following features:
    1) movie title embedding
    2) movie title text encoding using a GRU network
    3) movie genre text embedding
    """

    def __init__(self, movies_df, embedding_size=32, use_gru=False,
                 use_genres=False, layer_sizes=[64], debug=False,
                 max_num_movies=20_000):
        super().__init__()
        # option debug is for test using TF movielens-100k dataset
        self.debug = debug
        if self.debug:
            self.use_genres = False
            self.use_gru = False
        else:
            self.use_genres = use_genres
            self.use_gru = use_gru

        # movie title embedding
        max_tokens = 10_000
        self.movie_title_embedding = tf.keras.Sequential([
            tf.keras.layers.Hashing(num_bins=max_num_movies),
            tf.keras.layers.Embedding(
                input_dim=max_num_movies, output_dim=embedding_size)
        ])

        # movie title text encoding
        self.title_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens)
        if self.use_gru:
            self.movie_title_encoding = tf.keras.Sequential([
                self.title_vectorizer,
                tf.keras.layers.Embedding(
                    max_tokens, embedding_size, mask_zero=True),
                tf.keras.layers.GRU(embedding_size, return_sequences=True),
                tf.keras.layers.GRU(embedding_size, return_sequences=True),
                tf.keras.layers.GRU(embedding_size)
            ])
        else:
            self.movie_title_encoding = tf.keras.Sequential([
                self.title_vectorizer,
                tf.keras.layers.Embedding(
                    max_tokens, embedding_size, mask_zero=True),
                tf.keras.layers.GlobalAveragePooling1D()
            ])

        if self.debug:
            # test for TF dataset
            self.title_vectorizer.adapt(movies_df)
        else:
            self.title_vectorizer.adapt(movies_df['title'])

        if self.use_genres:
            # movie genres text encoding
            max_tokens = 100
            self.genre_vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=max_tokens)
            self.movie_genre_encoding = tf.keras.Sequential([
                self.genre_vectorizer,
                tf.keras.layers.Embedding(
                    max_tokens, embedding_size, mask_zero=True),
                tf.keras.layers.GlobalAveragePooling1D()
            ])
            self.genre_vectorizer.adapt(movies_df['genres'])

        # add deep dense layers
        self.dense_layers = tf.keras.Sequential()
        for ls in layer_sizes:
            self.dense_layers.add(tf.keras.layers.Dense(ls, activation='relu'))
        self.dense_layers.add(tf.keras.layers.Dense(embedding_size))

    def call(self, inputs):
        # for TF dataset test
        if self.debug:
            movie_title_emb = self.movie_title_embedding(inputs)
            title_text_emb = self.movie_title_encoding(inputs)
            movie_embeddings = tf.concat(
                [movie_title_emb, title_text_emb], axis=1)
        else:
            movie_title_emb = self.movie_title_embedding(inputs['title'])
            title_text_emb = self.movie_title_encoding(inputs['title'])
            if self.use_genres:
                genre_text_emb = self.movie_genre_encoding(inputs['genres'])
                movie_embeddings = tf.concat(
                    [movie_title_emb, title_text_emb, genre_text_emb], axis=1)
            else:
                movie_embeddings = tf.concat(
                    [movie_title_emb, title_text_emb], axis=1)

        # pass through dense layers
        return self.dense_layers(movie_embeddings)


class RetrievalModel(tfrs.models.Model):
    """
    This is the combined model of userModel and movieModel
    """

    def __init__(self, candidate_movies, timestamps, movies_df,
                 use_time=None, use_gru=False, use_genres=False,
                 embedding_size=32, time_encode_size=8, layer_sizes=[64],
                 debug=False, max_num_users=1_000, max_num_movies=10_000):
        super().__init__()
        self.debug = debug
        # query model
        self.query_model = UserModel(timestamps,
                                     emb_size=embedding_size,
                                     time_encode_size=time_encode_size,
                                     use_time=use_time,
                                     layer_sizes=layer_sizes,
                                     max_num_users=max_num_users)
        # condidate model
        self.candidate_model = MovieModel(movies_df,
                                          embedding_size=embedding_size,
                                          use_gru=use_gru,
                                          use_genres=use_genres,
                                          layer_sizes=layer_sizes,
                                          debug=debug,
                                          max_num_movies=max_num_movies)
        # define the retrieval task
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_movies.batch(
                    128).map(self.candidate_model)
            )
        )

    def compute_loss(self, features, training=False):
        # for TF dataset test
        if self.debug:
            query_embeddings = self.query_model({
                'userId': features['userId'],
                'timestamp': features['timestamp']
            })
            movie_embeddings = self.candidate_model({
                'title': features['title']
            })
        else:
            query_embeddings = self.query_model({
                'userId': features['userId'],
                'timestamp': features['timestamp'],
                'year': features['year'],
                'month': features['month'],
                'weekday': features['weekday'],
                'day': features['day'],
                'hour': features['hour']
            })
            movie_embeddings = self.candidate_model({
                'title': features['title'],
                'genres': features['genres'],
            })

        return self.task(query_embeddings, movie_embeddings, compute_metrics=not training)


class RatingModel(tf.keras.Model):
    """
    Class for predicting user ratings.
    """

    def __init__(self, timestamps, movies_df,
                 use_time=None, use_gru=False, use_genres=False,
                 embedding_size=32, time_encode_size=32, layer_sizes=[64],
                 debug=False, max_num_users=1_000, max_num_movies=10_000):
        super().__init__()
        self.debug = debug
        # user embedding
        self.user_embedding = UserModel(timestamps,
                                        emb_size=embedding_size,
                                        time_encode_size=time_encode_size,
                                        use_time=use_time,
                                        layer_sizes=layer_sizes,
                                        max_num_users=max_num_users)
        # movie embedding
        self.movie_embedding = MovieModel(movies_df,
                                          embedding_size=embedding_size,
                                          use_gru=use_gru,
                                          use_genres=use_genres,
                                          layer_sizes=layer_sizes,
                                          debug=debug,
                                          max_num_movies=max_num_movies)

        # predicting ratings
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, features):
        # for TF dataset test
        if self.debug:
            user_embeddings = self.user_embedding({
                'userId': features['userId'],
                'timestamp': features['timestamp']
            })
            movie_embeddings = self.movie_embedding({
                'title': features['title']
            })
        else:
            user_embeddings = self.user_embedding({
                'userId': features['userId'],
                'timestamp': features['timestamp'],
                'year': features['year'],
                'month': features['month'],
                'weekday': features['weekday'],
                'day': features['day'],
                'hour': features['hour']
            })
            movie_embeddings = self.movie_embedding({
                'title': features['title'],
                'genres': features['genres'],
            })
        embeddings = tf.concat([user_embeddings, movie_embeddings], axis=1)
        return self.ratings(embeddings)


class RankingModel(tfrs.models.Model):
    """
    Class for ranking model which fine tunes the output of RetrievalModel.
    """

    def __init__(self, timestamps, movies_df,
                 use_time=None, use_gru=False, use_genres=False,
                 embedding_size=32, time_encode_size=32, layer_sizes=[64],
                 debug=False, max_num_users=1_000, max_num_movies=10_000):
        super().__init__()
        self.rating_model = RatingModel(timestamps, movies_df,
                                        use_time=use_time,
                                        use_gru=use_gru,
                                        use_genres=use_genres,
                                        embedding_size=embedding_size,
                                        time_encode_size=time_encode_size,
                                        layer_sizes=layer_sizes,
                                        debug=debug,
                                        max_num_users=max_num_users,
                                        max_num_movies=max_num_movies
                                        )
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features):
        return self.rating_model(features)

    def compute_loss(self, features, training=False):
        labels = features.pop('rating')
        ratings_pred = self(features)
        return self.task(labels=labels, predictions=ratings_pred)
