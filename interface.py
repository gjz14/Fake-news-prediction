
# coding: utf-8

# In[1]:

import sys
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
    file.close();

    file = open('./modeldata/cvec.txt', 'rb')
    cvec = pickle.load(file)
    file.close();

    file = open('./modeldata/feature_weight_dict.txt', 'rb')
    feature_weight_dict = pickle.load(file)
    file.close();
    
    if len(sys.argv)==2:
        sample = sys.argv[1]
        class_names = ['TheOnion', 'nottheonion']
        word_weight_dict = Fake_master.my_analysis(nb, cvec, sample, feature_weight_dict)
        Fake_master.plot_myanalysis(word_weight_dict)

        Fake_master.prediction(sample,nb,cvec)
        Fake_master.lime_analysis(nb, cvec,sample)
        driver = webdriver.Chrome(ChromeDriverManager().install())
        driver.get("C:/Users/54440/Desktop/cse256/Final_project/Fake_oi.html")
        #ele = driver.findElement(By.id("top_divVI57UDM0NL5INZZ"));
        driver.execute_script(open("./myjs_script0.js").read());
       # driver.execute_script("alert(\"hahaha\")");


# In[2]:


#sample = "This Man Is Pretending To Piss For A Little Longer At The Urinal So He Doesn’t Have To Talk To One Of His Coworkers Washing His Hands"



