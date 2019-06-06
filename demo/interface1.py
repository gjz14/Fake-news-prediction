# coding: utf-8

# In[1]:

import sys
import Fake_master
import pickle
from sentiment.utility import generate_dynamic_analysis, generate_lime
from selenium.webdriver.common.keys import Keys

from selenium import webdriver
from nltk.tokenize import word_tokenize

# Fake_master.main()


# In[ ]:

def app(sentence, num):
    file = open('./model1_data/nb.txt', 'rb')
    nb = pickle.load(file)
    file.close()

    file = open('./model1_data/cvec.txt', 'rb')
    cvec = pickle.load(file)
    file.close()

    file = open('./model1_data/feature_weight_dict.txt', 'rb')
    feature_weight_dict = pickle.load(file)
    file.close()


    sample = sentence
    num_features = num
    class_names = ['TheOnion', 'nottheonion']

    '''
    word_weight_dict = Fake_master.my_analysis(nb, cvec, sample, feature_weight_dict)
    hasdict = Fake_master.plot_myanalysis(word_weight_dict)
    '''
    figpath = generate_dynamic_analysis(nb, cvec, 'fakenews', class_names, sample)
    # fig path is the image path for dynamic analysis images
    # Fake_master.prediction(sample, nb, cvec)
    limepath, piepath = generate_lime(nb, cvec, class_names, sample, 'TheOnion', 'fakenews', num_features)
    # lime path is the image path for lime analysis images

    return {'myanalysis_figpath': figpath, 'limepath': limepath, 'piepath': piepath}





if __name__ == "__main__":
    app()