from flask import Flask
from flask_restful import reqparse, Resource, Api
import pickle
import numpy as np
from model import Content_based, Collab_based, Hybrid_based, Model_based

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

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('film', required=False, help="film title")
parser.add_argument('limit', required=False,  help="N in top N films")
parser.add_argument('user', type=int, help='a user that exists in the list of users!')


class Film_all(Resource):
    def get(self):

        args = parser.parse_args()
        query = args['film']
        n= int(args['limit'])
        if query in movie_map.keys():
            clf1 = Content_based(latent_content)
            clf2 = Collab_based(latent_collaborative)
            clf3 = Hybrid_based(latent_content, latent_collaborative)
            output =  {
                    'content': {'content':clf1.predict_Top_N(query, n)},
                    'collaborative': {'collaborative':clf2.predict_Top_N(query, n)},
                    'hybrid': {'hybrid':clf3.predict_Top_N(query, n)},
                      }
        else:
            output = None
        return output


class Film(Resource):
    def get(self, basis_id):
        return basis_id


class User_recom(Resource):
    def get(self):
        args = parser.parse_args()
        ui = int(args['user'])
        if ui in rating.userId.unique():
            ui_list = rating[rating.userId == ui].movieId.tolist()
            d = {k: v for k,v in movie_map.items() if not v in ui_list}
            clf = Model_based(Algo)
            output = clf.predict_Top_N_user(ui, d, 20)

        else:
            output = None

        return output

api.add_resource(Film_all, '/')
api.add_resource(Film, '/<basis_id>')

api.add_resource(User_recom, '/user')

if __name__ == '__main__':
    app.run(debug=True, port=8000)


