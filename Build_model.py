# ML imports
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from surprise import Dataset, Reader, accuracy, SVD
from surprise.model_selection import train_test_split
import sys
#Load movies data
movies = pd.read_csv('ml-20m/movies.csv')
tags = pd.read_csv('ml-20m/tags.csv')
ratings = pd.read_csv('ml-20m/ratings.csv')

#manipulation of data frames for further detail see the jupyter notebook
movies['genres'] = movies['genres'].str.replace('|',' ')

#limit ratings to user ratings that have rated more that 55 movies
#it also filters the number of movies we can keep-- the reason is my
#laptop limited power.
ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 55)
movie_list_rating = ratings_f.movieId.unique().tolist()
# filter the movies data frame
movies = movies[movies.movieId.isin(movie_list_rating)]
#map movie to id:
Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))
#save user ratings
ratings_f.to_pickle('./Files/rating.pkl')
# pickle the map and id files:

with open('./Files/map.pkl', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(Mapping_file, f, pickle.HIGHEST_PROTOCOL)

# with open('./Files/Users.pkl', 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(Users, f, pickle.HIGHEST_PROTOCOL)

#remove unnecessary timesteps
tags.drop(['timestamp'],1, inplace=True)
ratings_f.drop(['timestamp'],1, inplace=True)
# ---------------------------------
# trying to make a content filter:
#make a useful data frame from tags and movies
mixed = pd.merge(movies, tags, on='movieId', how='left')

#create corpus from tags and genres
mixed.fillna("", inplace=True)
mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))
Final = pd.merge(movies, mixed, on='movieId', how='left')
Final['Corpus'] = Final[['tag', 'genres']].apply(lambda x: ' '.join(x), axis = 1)

#text transformation and and SVD to create latent matrix:
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(Final['Corpus'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=Final.index.tolist())
svd = TruncatedSVD(n_components=200)
latent_matrix_1 = svd.fit_transform(tfidf_df)
latent_matrix_1_df = pd.DataFrame(latent_matrix_1, index=Final.title.tolist())
#content filter latent matrix data frame
latent_matrix_1_df.to_pickle('./Files/latent_content.pkl')
# ----------------------------------
# now collaborative item-item filter:
ratings_f1 = pd.merge(movies['movieId'], ratings_f, on="movieId", how="right")
ratings_f2 = ratings_f1.pivot(index = 'movieId', columns ='userId', values = 'rating').fillna(0)
svd = TruncatedSVD(n_components=200)
latent_matrix_2 = svd.fit_transform(ratings_f2)
latent_matrix_2_df = pd.DataFrame(latent_matrix_2, index=Final.title.tolist())
#collabortive filter latent matrix data frame
latent_matrix_2_df.to_pickle('./Files/latent_collaborative.pkl')
# ----------------------------------
# now a user collabortive model using Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_f1[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25)
# We'll use the famous SVD algorithm.
algorithm = SVD()
# Train the algorithm on the trainset, and predict ratings for the testset
algorithm.fit(trainset)
# Then compute RMSE
accuracy.rmse(algorithm.test(testset))
# save model to file
with open('./Files/model_svd.pkl', 'wb') as f:
    pickle.dump(algorithm, f, pickle.HIGHEST_PROTOCOL)

