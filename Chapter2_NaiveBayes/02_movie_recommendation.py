"""Implementing a movie recommendation system using the Naive Bayes
Algorithm.
@date: 24/03/2023
@author: Víctor Bocanegra

The data we're working with was obtained from:
https://grouplens.org/datasets/movielens/: ml-latest-small.zip (size: 1 MB)

"""

import numpy as np
import pandas as pd
from collections import defaultdict

data_path = "./datasets/ml-latest-small/ratings.csv"


def count_unique_ids(data_path):
	"""
	@param data_path: data_path (str): The path to the CSV containing the data.
	@return: tuple: A tuple containing two integers, the first one representing
		the count of unique user IDS and the second one representing the
		unique number of movieIds.
	"""
	# Read the csv file
	data = pd.read_csv(data_path)

	# Count the unique userIds and movieIds
	unique_user_ids = data['userId'].nunique()
	unique_movie_ids = data['movieId'].nunique()

	return unique_user_ids, unique_movie_ids


def load_rating_data(data_path, n_users, n_movies):
	"""
	Load rating data from file and also return the number of ratings for each movie and movie_id index mapping.
	@param data_path: Path to the rating data file
	@param n_users: number of users
	@param n_movies: number of movies that have ratings
	@return: rating data in the numpy array of [user, movie];
			movie_n_rating, {movie_id: number of ratings};
			movie_id_mapping, {movie_id: column index in rating data}
	"""
	data = np.zeros([n_users, n_movies], dtype = np.float32)
	movie_id_mapping = {}
	movie_n_rating = defaultdict(int)
	with open(data_path, 'r') as file:
		for line in file.readlines()[1:]:
			user_id, movie_id, rating, _ = line.split(",")
			user_id = int(user_id) - 1
			if movie_id not in movie_id_mapping:
				movie_id_mapping[movie_id] = len(movie_id_mapping)
			rating = int(float(rating))
			data[user_id, movie_id_mapping[movie_id]] = rating
			if rating > 0:
				movie_n_rating[movie_id] += 1
	return data, movie_n_rating, movie_id_mapping


# Analyzing the data distribution
def display_distribution(data):
	values, counts = np.unique(data, return_counts = True)
	print("Checking value, counts: \n")
	print(values, counts)
	for value, count in zip(values, counts):
		print(f"Number of rating {int(value)}: {count}")



n_users, n_movies = count_unique_ids(data_path)
data, movie_n_rating, movie_id_mapping = load_rating_data(data_path, n_users, n_movies)

display_distribution(data)


