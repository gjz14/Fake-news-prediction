

import sys
import Fake_master
import pickle
from sentiment.utility import generate_dynamic_analysis, generate_lime

from nltk.tokenize import word_tokenize

def app(sentence, num):
    file = open('./model1_data/sentiment_model.txt', 'rb')
    nb = pickle.load(file)
    file.close()

    file = open('./model1_data/sentiment_vect.txt', 'rb')
    cvec = pickle.load(file)
    file.close()


    sample = sentence
    num_features = num
    class_names = ['NEGATIVE', 'POSITIVE']

    figpath = generate_dynamic_analysis(nb, cvec, 'sentiment', class_names, sample)
    # fig path is the image path for dynamic analysis images

    limepath, piepath = generate_lime(nb, cvec, class_names, sample, 'fuck', 'sentiment', num_features)
    # lime path is the image path for lime analysis images
    return {}

if __name__ == "__main__":
     app()
