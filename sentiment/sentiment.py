import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
import utility

def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
                
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    
    sentiment.vect = TfidfVectorizer()
    sentiment.trainX = sentiment.vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.vect.transform(sentiment.dev_data)
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

if __name__ == "__main__":
    print("Reading data")
    tarfname = "./data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    print("\nTraining classifier")

    cls = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000, C=2.97)
    cls.fit(sentiment.trainX, sentiment.trainy)
    yp = cls.predict(sentiment.devX)
    acc = metrics.accuracy_score(sentiment.devy, yp)
    print("Accuracy: ", acc)
    
    utility.generate_wordcloud(cls, sentiment.vect, 10, "sentiment")
    class_names = ['NEGATIVE', 'POSITIVE']
    utility.generate_lime(cls, sentiment.vect, class_names, sentiment.dev_data[0], sentiment.dev_labels[0], "sentiment")
        
    # for idx, (x, y) in enumerate(zip(sentiment.devy, yp)):
    #     if x != y:
    #         exp = explainer.explain_instance(sentiment.dev_data[idx], c.predict_proba, num_features=6)
    #         print('Document id: %d' % idx)
    #         print(sentiment.dev_data[idx])
    #         print('Probability(pos) =', c.predict_proba([sentiment.dev_data[idx]])[0,1])
    #         print('Prediction: ', class_names[c.predict([sentiment.dev_data[idx]])[0]])
    #         print('True class: %s' % sentiment.dev_labels[idx])
    #         print(exp.as_list())