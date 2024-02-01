import numpy as np
import tensorflow as tf
import pandas as pd
import tqdm

def load_precalc_params(Y, num_features):
    """
    Args:
      Y_df (pandas.DataFrame): movie ratings (movieId, movie ratings for userIds)
    Returns:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters (biases)
      num_movies (int)                    : number of movies
      num_users (int)                     : number of users
    """
    num_movies, num_users = Y.shape
    X = np.random.rand(num_movies, num_features)
    W = np.random.rand(num_users, num_features)
    b = np.random.rand(1, num_users)
    return X, W, b, num_movies, num_users

def load_ratings_small(path, count=1000):
    """
      Args:
        path (str): path to the folder where the MovieLens (small) is located
      Returns:
        Y_df (pandas.DataFrame): movie ratings (movieId, movie ratings for userIds)
        R_df (pandas.DataFrame): rating flags (movieId, movie rating flags for userIds)
         """
    ratings_df = pd.read_csv(path + '/ratings.csv')[:count]
    movies_df = pd.read_csv(path + '/movies.csv').drop_duplicates('movieId')
    user_ids = ratings_df['userId'].unique().tolist()
    columns = ['movieId', 'title'] + user_ids
    Y_df = pd.DataFrame(columns=columns, data=np.zeros((movies_df.shape[0], len(columns))))
    R_df = pd.DataFrame(columns=columns, data=np.zeros((movies_df.shape[0], len(columns))))
    Y_df[['movieId', 'title']], R_df[['movieId', 'title']] = movies_df[['movieId', 'title']], movies_df[['movieId', 'title']]
    for row in tqdm.tqdm(ratings_df.itertuples()):
        Y_df.loc[Y_df['movieId'] == row.movieId, row.userId] = row.rating
        R_df.loc[R_df['movieId'] == row.movieId, row.userId] = 1
    return Y_df, R_df

def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0
    _, k = W.shape
    for j in range(nu):
        for i in range(nm):
            J += R[i][j] * np.power(np.dot(W[j], X[i]) + b[0][j] - Y[i][j], 2)
    Rw, Rx = 0, 0
    for j in range(nu):
        Rw += np.sum(np.power(W[j], 2))
    for i in range(nm):
        Rx += np.sum(np.power(X[i], 2))
    J = J / 2 + (lambda_ / 2) * (Rw + Rx)
    return J

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

def normalize_ratings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    Args:
      Y (ndarray (num_movies, num_users)): matrix of user ratings of movies
      R (ndarray (num_movies, num_users)): matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
    Returns:
      Ynorm (ndarray (num_movies, num_users)): matrix of normalized user ratings of movies
      Ymean (ndarray (num_movies, 1)): vector of mean rating for movies
    """
    m, n = Y.shape
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = np.where(R[i, :] == 1)
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm, Ymean