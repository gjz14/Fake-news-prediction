import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from nltk.tokenize import word_tokenize
import os

def generate_dynamic_analysis(model, vectorizier, out_name, class_names, text):
    
    feature_names = vectorizier.get_feature_names()
    feature_weights = model.coef_[0]
    feature_weight_dict = {}
    for i in range(len(feature_names)):
        feature_weight_dict[feature_names[i]] = feature_weights[i]
        
   # class_names = ['TheOnion', 'nottheonion']
    word_list = word_tokenize(text)
    for i in range(len(word_list)):
        word_list[i] = word_list[i].lower()
    word_weight_dict = {}
    for w in word_list:
        if w in feature_weight_dict.keys():
            word_weight_dict[w] = feature_weight_dict[w]
            
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
    path = os.path.join("./img", out_name + "_dynamic.png")
    plt.savefig(path)
    return True
            
def generate_wordcloud(model, vectorizier, k, out_name):
    coefficients = model.coef_[0]
    k = 8
    top_k =np.argsort(coefficients)[-k:]
    top_k_words = []

    print('-'*50)
    print('Top k=%d' %k)
    print('-'*50)

    for i in top_k:
        # print(vectorizier.get_feature_names()[i])
        top_k_words.append(vectorizier.get_feature_names()[i])
    print(top_k_words)
    print('-'*50)
    print('Bottom k=%d' %k)
    print('-'*50)
    bottom_k =np.argsort(coefficients)[:k]
    bottom_k_words = []
    for i in bottom_k:
        # print(vectorizier.get_feature_names()[i])
        bottom_k_words.append(vectorizier.get_feature_names()[i])
    print(bottom_k_words)
    print("\n")
    top_path = os.path.join("./img", out_name + "_top_k.png")
    bottom_path = os.path.join("./img", out_name + "_bottom_k.png")
    top_k_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(" ".join(top_k_words))
    top_k_wordcloud.to_file(top_path)
    bottom_k_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(" ".join(bottom_k_words))
    bottom_k_wordcloud.to_file(bottom_path)
    # plt.figure()
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()

def generate_lime(model, vectorizier, class_names, text, true_label, out_name):
    c = make_pipeline(vectorizier, model)
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, c.predict_proba, num_features=6)
    print('Input text: ', text)
    print('Probability(%s) =' % class_names[1], c.predict_proba([text])[0, 1])
    print('Prediction: ', class_names[c.predict([text])[0]])
    print('True class: %s' % true_label)

    path_name = os.path.join("../lime", out_name)
    exp.save_to_file(path_name + ".html")
    # exp.as_pyplot_figure().savefig(path_name + ".jpg")
    # fig = exp.as_pyplot_figure()
    # plt.show()