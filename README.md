# Movie Recommender Systems

## Introduction

An AI-based recommender system is now an integrated part of many online service platforms including retails (e.g., Amazon), entertainment(e.g., Netflix and Spotify), publishing (e.g., scientific journals and magazines), social media (e.g., LinkedIn and Twitter), and so on. A major motivation for companies to deploy a recommender system is to increase revenues by personalized services. By effectively recommending the most relevant and accurate items that align with customers’ needs and interests, a recommender system can influence customers' decisions (both qualitatively and quantitatively) toward higher customer retention rate and more revenues. For example, two widely cited successes[^1] are a) 35% of Amazon’s sales was achieved by recommendation system; b) 75% of watched content on Netflix came from recommendation system. Personalized recommendation is expected to play an increasingly important role in various sectors of business, as both the product base and user base continue to increase and diversify. A significant part of the future recommendation landscape can be content recommendation such as movies, songs, articles, news, etc. This project will build an AI-based movie recommender system that can provide highly personalized and efficient content recommendation.

Depending on the scenarios and needs, different methods can be used in the recommender system. For example:

* [Global recommendation](./Global_recommendation.ipynb)
* [Content based recommendation](./Content_based_recommendation.ipynb)
* [Collaborative filtering recommendation](./Collaborative_filtering_recommendation.ipynb)
* [Deep learning hybrid recommender](./Deep_learning_hybrid_model.ipynb)

See the corresponding notebooks in this repo for details. Here, a modern recommendation system based on [Tensorflow Recommender](https://www.tensorflow.org/recommenders) is introduced.

## Framework

A typical recommendation process can consist of two stages: 

1) The retrieval stage. By efficient initial screening, a set of hundreds of candidates can be selected from all possible candidates. The main objective is to efficient weed out all candidates that are not interesting to a user.

2) The ranking stage. Based on the results from retrieval stage, furhter fine tuning can be made to make a shortlist of most likely candidates.

## Dataset

In this example, the [Movielens latest-small dataset](https://grouplens.org/datasets/movielens/latest/) is used to implement a modern movie recommender system. A detailed exploratory data analysis can be found in this [notebook](./Exploratory_data_analysis.ipynb).

Several crucial preprocessing is outlined here:

1) It turns out that excluding those movies with number of ratings less than 20 (or a similar value) is beneficial for model training/validation.

2) Extracting data such as year, month, weekday, day, hours from timestamps can provide useful context information.

3) Timestamp, if used as continous quantity, should be normalized.

4) Remove any NaN rows.

5) Need to merge ratings and movies dataframes to get unified training examples. 

6) Split into Train/Test dataset before training. 

See the Data Preprocessing section in [this notebook](./movie_recommender.ipynb) for details. 

## Retrieval Model 

Retrieval model is a two-tower architecture consisting of a user model and movie model. The user model can take advantages of various embeddings such as 
1) user id embedding
2) discretized timestamp embedding
3) cyclic data encoding (e.g., using sine/cosine encoding)
4) normalized continuous timestamp quantatity

The movie model can also be constructed using
1) movie title embedding after text vectorization
2) movie title text encoding using GRU or a 1D global average pooling
3) movie genres embedding.

The training of retrieval model utilizes data in terms of implicit feedback, i.e., which movies the users have watched/rated and which they did not. The retrieval model is trained to recognize the affinity between a user and a movie. Such affinity scores can be effectively measured by the [tfrs.metrics.FactorizedTopK](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/metrics/FactorizedTopK) metric. 

## Ranking Model

A typical ranking model also consists of a user model (to learn user representation) and a movie model (to learn movie representation). The above mentioned embeddings and data transformations can be applied here as well. However, the ranking model directly predicts user ratings on movies. Such predicted ratings can be utilized to make a shortlist from the retrieval results.

## Results

Various models have been implemented in the [model.py](./models.py) file. An end-to-end data preparation, model building, model training/validation, and model serving can be found in this [notebook](./movie_recommender.ipynb).



[^1]: https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers

