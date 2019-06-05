import sys
import pickle
import numpy as np
from model import memory_based, model_based

with open('./Files/model_svd.pkl', 'rb') as f:
    Algo = pickle.load(f)
with open('./Files/map.pkl', 'rb') as f:
    movie_map = pickle.load(f)
with open('./Files/rating.pkl', 'rb') as f:
    rating = pickle.load(f)
with open('./Files/latent_collaborative.pkl', 'rb') as f:
    latent_collaborative = pickle.load(f)
with open('./Files/latent_content.pkl', 'rb') as f:
    latent_content = pickle.load(f)

#model = MODEL(latent_content, latent_collaborative, movie_map, rating, Algo)
query = 'Strada, La (1954)'

if query in movie_map.keys():
    clf = memory_based(latent_content, latent_collaborative)
    output = clf.predict_Top_N_(query, 'hybrid', 10)
    print(output)
else:
    print('The movie does not exist in the database!')
    sys.exit()

ui = 13
if ui in rating.userId.unique():
    ui_list = rating[rating.userId == ui].movieId.tolist()
    d = {k: v for k,v in movie_map.items() if not v in ui_list}
    clf = model_based(Algo)
    output = clf.predict_Top_N_user(ui, d, 20)
    print(output)
else:
    print("User Id does not exist in the list!")
    sys.exit()

#res = model.Pred_similar('Strada, La (1954)','hybrid')
#model.Pred_user(11)
#print(res)

