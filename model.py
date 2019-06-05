import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class Content_based(object):

    def __init__(self, latent1):
        """top N memory based: content
        """
        self.content = latent1

    def predict_Top_N(self, query, n=10):
        a_1 = np.array(self.content.loc[query]).reshape(1, -1)
        content = cosine_similarity(self.content, a_1).reshape(-1)
        dictDf = {'content': content }
        similar = pd.DataFrame(dictDf, index = self.content.index )
        similar.sort_values('content', ascending=False, inplace=True)
        return similar.head(n+1)[1:].index.tolist()


class Collab_based(object):

    def __init__(self, latent2):
        """top N memory based: collaborative
        """
        self.collab = latent2

    def predict_Top_N(self, query, n=10):

        a_1 = np.array(self.collab.loc[query]).reshape(1, -1)
        collab = cosine_similarity(self.collab, a_1).reshape(-1)
        dictDf = {'collaborative': collab }
        similar = pd.DataFrame(dictDf, index = self.collab.index )
        similar.sort_values('collaborative', ascending=False, inplace=True)
        return similar.head(n+1)[1:].index.tolist()


class Hybrid_based(object):

    def __init__(self, latent1, latent2):
        """top N memory based: hybrid
        """
        self.content = latent1
        self.collab = latent2

    def predict_Top_N(self, query, n=10):

        a_1 = np.array(self.content.loc[query]).reshape(1, -1)
        a_2 = np.array(self.collab.loc[query]).reshape(1, -1)
        content = cosine_similarity(self.content, a_1).reshape(-1)
        collaborative = cosine_similarity(self.collab, a_2).reshape(-1)
        hybrid = ((content + collaborative)/2.0)
        # a data frame of movies based on similarity to query
        dictDf = {'hybrid': hybrid }
        similar = pd.DataFrame(dictDf, index = self.content.index )
        similar.sort_values('hybrid', ascending=False, inplace=True)
        return similar.head(n+1)[1:].index.tolist()


class Model_based(object):

    def __init__(self, algorithm):
        """top N for a particular user
        """
        self.algo = algorithm

    def predict_Top_N_user(self, ui, d, n=10):
        predictedL = []
        for i, j in d.items():
            predicted = self.algo.predict(ui, j)
            predictedL.append((i, predicted[3]))
        pdf = pd.DataFrame(predictedL, columns = ['movies', 'ratings'])
        pdf.sort_values('ratings', ascending=False, inplace=True)
        pdf.set_index('movies', inplace=True)
        return pdf.head(n).index.tolist()







