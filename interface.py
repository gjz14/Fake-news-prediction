
# coding: utf-8

# In[1]:

import sys
sys.path.insert(0, './sentiment/')

import utility
import Fake_master
import pickle
from selenium.webdriver.common.keys import Keys


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from nltk.tokenize import word_tokenize
#Fake_master.main()


# In[ ]:


if __name__ == "__main__":
    file = open('./modeldata/nb.txt', 'rb')
    nb = pickle.load(file)
    file.close()

    file = open('./modeldata/cvec.txt', 'rb')
    cvec = pickle.load(file)
    file.close()

    file = open('./modeldata/feature_weight_dict.txt', 'rb')
    feature_weight_dict = pickle.load(file)
    file.close()
    
    if len(sys.argv)==3:
        sample = sys.argv[1]
        num_features = sys.argv[2]
        class_names = ['TheOnion', 'nottheonion']
        
        '''
        word_weight_dict = Fake_master.my_analysis(nb, cvec, sample, feature_weight_dict)
        hasdict = Fake_master.plot_myanalysis(word_weight_dict)
        '''
        figpath = utility.generate_dynamic_analysis(nb, cvec, 'fakenews', class_names, sample)
        # fig path is the image path for dynamic analysis images
        Fake_master.prediction(sample,nb,cvec)
        limepath, piepath = utility.generate_lime(nb, cvec, class_names, sample, 'TheOnion', 'fakenews', num_features)
        # lime path is the image path for lime analysis images
        
        return {'myanalysis_figpath':figpath,'limepath':limepath, 'piepath':piepath}


# In[2]:


#sample = "This Man Is Pretending To Piss For A Little Longer At The Urinal So He Doesn’t Have To Talk To One Of His Coworkers Washing His Hands"



