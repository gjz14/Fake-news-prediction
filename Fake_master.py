
# coding: utf-8

# In[2]:


# Basic libraries
import sys
sys.path.insert(0, './sentiment/')

import utility
import numpy as np
import pandas as pd
import pickle

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
#get_ipython().run_line_magic('matplotlib', 'inline')

# Natural Language Processing
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from nltk.tokenize import word_tokenize

# In[3]:


def read_data(dname):
    df =  pd.read_csv(dname)
    return df

def clean_data(dataframe):

    # Drop duplicate rows
    dataframe.drop_duplicates(subset='title', inplace=True)
    
    # Remove punctation
    dataframe['title'] = dataframe['title'].str.replace('[^\w\s]',' ')

    # Remove numbers 
    dataframe['title'] = dataframe['title'].str.replace('[^A-Za-z]',' ')

    # Make sure any double-spaces are single 
    dataframe['title'] = dataframe['title'].str.replace('  ',' ')
    dataframe['title'] = dataframe['title'].str.replace('  ',' ')

    # Transform all text to lowercase
    dataframe['title'] = dataframe['title'].str.lower()
    
    #print("New shape:", dataframe.shape)
    return dataframe.head()

def preprocess(df):
    #Reset the index
    df = df.reset_index(drop=True)
    # Replace `TheOnion` with 1, `nottheonion` with 0
    df["subreddit"] = df["subreddit"].map({"nottheonion": 0, "TheOnion": 1})



def countvector_logisticregression(X_train,y_train,X_test,y_test):
    
    pipe = Pipeline([('cvec', CountVectorizer()),    
                 ('lr', LogisticRegression(solver='liblinear'))])

    # Tune GridSearchCV
    pipe_params = {'cvec__stop_words': [None, 'english'],
                   'cvec__ngram_range': [(1,1), (2,2), (1,3)],
                   'lr__C': [0.01, 1]}

    gs = GridSearchCV(pipe, param_grid=pipe_params, cv=3)
    gs.fit(X_train, y_train);
    print("Best score:", gs.best_score_)
    print("Train score", gs.score(X_train, y_train))
    print("Test score", gs.score(X_test, y_test))

    return gs


# In[4]:


def best_model(X_train,X_test,y_train,y_test):
    #Instantiate the classifier and vectorizer
    nb = MultinomialNB(alpha = 0.36)
    cvec = CountVectorizer(ngram_range= (1, 3))

    # Fit and transform the vectorizor
    cvec.fit(X_train)

    Xcvec_train = cvec.transform(X_train)
    Xcvec_test = cvec.transform(X_test)

    # Fit the classifier
    nb.fit(Xcvec_train,y_train)

    # Create the predictions for Y training data
    preds = nb.predict(Xcvec_test)

    print("The mean accuracy on the test set: "+ str(nb.score(Xcvec_test, y_test)))
    
    return nb,cvec

def best_model2(X_train,X_test,y_train,y_test):
    class_names = ['TheOnion', 'nottheonion']
    stop_words_onion = stop_words.ENGLISH_STOP_WORDS
    stop_words_onion = list(stop_words_onion)
    stop_words_onion.append('onion')
    #Instantiate the classifier and vectorizer
    lr = LogisticRegression(C = 1.0, solver='liblinear')
    cvec2 = CountVectorizer(stop_words = None)

    # Fit and transform the vectorizor
    cvec2.fit(X_train)

    Xcvec2_train = cvec2.transform(X_train)
    Xcvec2_test = cvec2.transform(X_test)

    # Fit the classifier
    lr.fit(Xcvec2_train,y_train)

    # Create the predictions for Y training data
    lr_preds = lr.predict(Xcvec2_test)
    '''
    print(len(lr_preds), len(y_test))
    for i in range(len(lr_preds.tolist())):
        if lr_preds.tolist()[i] != y_test.tolist()[i]:
            print(lr_preds.tolist()[i], y_test.tolist()[i])
            print(X_test.tolist()[i])
    '''
    print("The mean accuracy on the test set: "+ str(lr.score(Xcvec2_test, y_test)))
    utility.plot_confusion_matrix(y_test, lr_preds, class_names, "Fake_news")
    return lr, cvec2


def coefficient_analysis(nb,cvec):
    # Create list of logistic regression coefficients 
    lr_coef = np.array(nb.coef_).tolist()
    lr_coef = lr_coef[0]

    # create dataframe from lasso coef
    lr_coef = pd.DataFrame(np.round_(lr_coef, decimals=3), 
    cvec.get_feature_names(), columns = ["penalized_regression_coefficients"])

    # sort the values from high to low
    lr_coef = lr_coef.sort_values(by = 'penalized_regression_coefficients', 
    ascending = False)

    # Jasmine changing things up here on out! Top half not mine. 
    # create best and worst performing lasso coef dataframes
    df_head = lr_coef.head(10)
    df_tail = lr_coef.tail(10)

    # merge back together
    df_merged = pd.concat([df_head, df_tail], axis=0)

    # plot the sorted dataframe
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    fig.suptitle('Coefficients!', size=14)
    ax = sns.barplot(x = 'penalized_regression_coefficients', y= df_merged.index, 
    data=df_merged)
    ax.set(xlabel='Penalized Regression Coefficients')
    plt.tight_layout(pad=3, w_pad=0, h_pad=0);
    
    
    print("The word that contributes the most positively to being from r/TheOnion is", 
          df_merged.index[0], "followed by", 
          df_merged.index[1], "and",
          df_merged.index[2],".")

    print("-----------------------------------")

    print("The word that contributes the most positively to being from r/nottheonion is", 
          df_merged.index[-1], "followed by", 
          df_merged.index[-2], "and",
          df_merged.index[-3],".")
    
    # Show coefficients that affect r/TheOnion
    df_merged_head = df_merged.head(10)
    exp = df_merged_head['penalized_regression_coefficients'].apply(lambda x: np.exp(x))
    df_merged_head.insert(1, 'exp', exp)
    df_merged_head.sort_values('exp', ascending=False)
    
    print("As occurences of", df_merged_head.index[0], "increase by 1 in a title, that title is", 
      round(df_merged_head['exp'][0],2), "times as likely to be classified as r/TheOnion.")
    # Show coefficients that affect r/nottheonion
    df_merged_tail = df_merged.tail(10)
    exp = df_merged_tail['penalized_regression_coefficients'].apply(lambda x: np.exp(x * -1))
    df_merged_tail.insert(1, 'exp', exp)
    df_merged_tail.sort_values('exp', ascending=False)
    print("As occurences of", df_merged_tail.index[-1], "increase by 1 in a title, that title is", 
      round(df_merged_tail['exp'][-1],2), "times as likely to be classified as r/nottheonion.")


# In[5]:
def prediction(corpus, nb, cvec):
    # transform corpus to test df
    X_test = [corpus]
    Xcvec_test = cvec.transform(X_test)
    preds = nb.predict(Xcvec_test)
   # print(nb.coef_) 
    return preds

def lime_analysis(nb, cvec,sample):
    class_names = ['TheOnion', 'nottheonion']
    c = make_pipeline(cvec, nb)
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(sample, c.predict_proba, num_features=9)
    exp.save_to_file('Fake_oi.html')
    expmap = exp.as_list()
    figure = exp.as_pyplot_figure()
    myhtml = exp.as_html()
    #prediction(sample,nb,cvec)
    return expmap, figure, myhtml

def my_analysis(nb, cvec, sample, feature_weight_dict):
    class_names = ['TheOnion', 'nottheonion']
    word_list = word_tokenize(sample)
    for i in range(len(word_list)):
        word_list[i] = word_list[i].lower()
    word_weight_dict = {}
    for w in word_list:
        if w in feature_weight_dict.keys():
            word_weight_dict[w] = feature_weight_dict[w]
    return word_weight_dict

def plot_myanalysis(word_weight_dict):
    
    if(len(word_weight_dict)==0):
        return False
    # create dataframe from lasso coef
    df_merged = pd.DataFrame(list(word_weight_dict.values()), 
word_weight_dict.keys(), columns = ["penalized_regression_coefficients"])
    df_merged.sort_values('penalized_regression_coefficients', ascending=False)
    # plot the dataframe
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    fig.suptitle('Coefficients analysis on feature weights', size=14)
    ax = sns.barplot(x = 'penalized_regression_coefficients', y= df_merged.index, 
    data=df_merged)
    ax.set(xlabel='Penalized Regression Coefficients')
    plt.tight_layout(pad=3, w_pad=0, h_pad=0);
    # save my analysis to png file
    plt.savefig('testimage1.png')
    return True

def main():
    dnames = ['./data/the_onion.csv', './data/not_onion.csv']
    class_names = ['TheOnion', 'nottheonion']
    df_onion = read_data(dnames[0])
    df_not_onion = read_data(dnames[1])
    clean_data(df_onion)
    clean_data(df_not_onion)
    # Combine df_onion & df_not_onion with only 'subreddit' (target) and 'title' (predictor) columns
    df = pd.concat([df_onion[['subreddit', 'title']], df_not_onion[['subreddit', 'title']]], axis=0)
    preprocess(df)
    # Baseline score
    print("--------Baseline score--------------")
    print(df['subreddit'].value_counts(normalize=True))
    X = df['title']
    y = df['subreddit']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=42,
                                                    stratify=y)
    
    #gs = countvector_logisticregression(X_train,y_train,X_test,y_test)
    print("-------------Best Model score------------------")
    nb, cvec = best_model2(X_train,X_test,y_train,y_test)
    #coefficient_analysis(nb,cvec)
    feature_names = cvec.get_feature_names()
    feature_weights = nb.coef_[0]
    feature_weight_dict = {}
    for i in range(len(feature_names)):
        feature_weight_dict[feature_names[i]] = feature_weights[i]
    file1 = open('nb.txt', 'wb')
    pickle.dump(nb, file1)
    file2 = open('cvec.txt', 'wb')
    pickle.dump(cvec, file2)
    file3 = open('feature_weight_dict.txt', 'wb')
    pickle.dump(feature_weight_dict, file3)
    file1.close()
    file2.close()
    file3.close()
    utility.generate_wordcloud(nb, cvec, 10, "Fake_news")
    return nb, cvec, feature_weight_dict
    
if __name__ == "__main__":
    nb, cvec, feature_weight_dict = main()
    
    if len(sys.argv)==2:
        print("------------Prediction of the input argv---------")
        pred = prediction(sys.argv[1],nb,cvec)
        print(pred)
        expmap, figure, myhtml = lime_analysis(nb, cvec, sys.argv[1])
        print(expmap)
        figure.show()
        #print(myhtml)
        word_weight_dict = my_analysis(nb, cvec, sys.argv[1], feature_weight_dict)
        plot_myanalysis(word_weight_dict)

