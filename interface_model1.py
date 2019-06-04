import sys
sys.path.insert(0, './sentiment/')

import utility
import pickle

from nltk.tokenize import word_tokenize

if __name__ == "__main__":
    file = open('./modeldata/sentiment.txt', 'rb')
    nb = pickle.load(file)
    file.close()

    file = open('./modeldata/sentiment_vect.txt', 'rb')
    cvec = pickle.load(file)
    file.close()

    
    if len(sys.argv)==3:
        sample = sys.argv[1]
        num_features = sys.argv[2]
        class_names = ['NEGATIVE', 'POSITIVE']

        figpath = utility.generate_dynamic_analysis(nb, cvec, 'sentiment', class_names, sample)
        # fig path is the image path for dynamic analysis images
        
        limepath, piepath = utility.generate_lime(nb, cvec, class_names, sample, 'sentiment', num_features)
        # lime path is the image path for lime analysis images    
