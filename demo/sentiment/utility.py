import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
import os

def generate_dynamic_analysis(model, vectorizier, out_name, class_names, text):
    
    feature_names = vectorizier.get_feature_names()
    feature_weights = model.coef_[0]
    feature_weight_dict = {}
    for i in range(len(feature_names)):
        feature_weight_dict[feature_names[i]] = feature_weights[i]
        
    word_list = word_tokenize(text)
    for i in range(len(word_list)):
        word_list[i] = word_list[i].lower()
    word_weight_dict = {}
    for w in word_list:
        if w in feature_weight_dict.keys():
            word_weight_dict[w] = feature_weight_dict[w]
            
    if(len(word_weight_dict) == 0):
        return False
    # create dataframe from lasso coef
    df_merged = pd.DataFrame(list(word_weight_dict.values()), word_weight_dict.keys(), columns = ["penalized_regression_coefficients"])
    df_merged.sort_values('penalized_regression_coefficients', ascending=False)
    # plot the dataframe
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    fig.suptitle('Coefficients analysis on feature weights', size=14)
    ax = sns.barplot(x = 'penalized_regression_coefficients', y= df_merged.index, data=df_merged)
    ax.set(xlabel='Penalized Regression Coefficients')
    plt.tight_layout(pad=3, w_pad=0, h_pad=0)
    # save my analysis to png file
    #path = os.path.join("./img", out_name + "_dynamic.png")
    
    # save to certain file
    path_name = os.path.join("./static/images/", out_name + "_dynamic.png")
    
    plt.savefig(path_name)
    plt.close()
    return path_name

def plot_confusion_matrix(y_true, y_pred, class_names, out_name):
    """
    Reference:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title = out_name + 'Confusion matrix'
            
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # plt.show()
    path = os.path.join("./img", out_name + "_confusion_matrix.png")
    plt.savefig(path)
    return ax
            
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

def generate_lime(model, vectorizier, class_names, text, true_label, out_name, num_features):
    '''
    model: the model used to train the data
    vectorizier: vectorizier used to vectorize data
    class_names: class of predictions
    text: input text
    true_label: actual label of the text
    out_name: output file name
    '''
    c = make_pipeline(vectorizier, model)
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, c.predict_proba, num_features=num_features)

    path_name = os.path.join("./static/images", out_name + "_lime.png")
    # exp.save_to_file(path_name + ".html")
    exp.as_pyplot_figure().savefig(path_name)
    # fig = exp.as_pyplot_figure()
    # plt.show()

    plt.close()
    # pie plot
    
    X_test = [text]
    Xcvec_test = vectorizier.transform(X_test)
    #class_names = ['TheOnion','nottheonion']
    probas = model.predict_proba(Xcvec_test)
    probas  = probas.tolist()[0]

    labels = class_names[0], class_names[1]
    sizes = [probas[0],probas[1]]
    colors = ['lightcoral', 'lightskyblue']
    explode = (0, 0)  # explode 1st slice



    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    #plt.show()
    
    # save to certain file
    path_name_pie = os.path.join("./static/images/", out_name + "_pie.png")
    plt.title('Prediction pie chart')
    plt.savefig(path_name_pie)
    plt.close()
    return path_name,path_name_pie