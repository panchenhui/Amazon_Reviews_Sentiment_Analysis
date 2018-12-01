import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

'''
TF-IDF
'''
# from sklearn.feature_extraction.text import TfidfVectorizer
# # read corpus
# corpus = []
# with open('reviews_Apps_for_Android_5.json') as c_f:
#     for line in c_f.readlines():
#         corpus.append(json.loads(line.strip())['reviewText'].lower())
#
# # Get IDF for each words over the corpus
# vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
# vectorizer.fit(corpus)
#
# train_X = []
# train_Y = []
# with open('data_training.json') as train_f:
#     for line in train_f.readlines():
#         dic = json.loads(line.strip())
#         train_X.append(vectorizer.transform([dic['data'].encode('utf-8').lower()])[0].toarray()[0])
#         train_Y.append(dic['label'])
#
# test_X = []
# test_Y = []
# with open('data_test.json') as test_f:
#     for line in test_f.readlines():
#         dic = json.loads(line.strip())
#         test_X .append(vectorizer.transform([dic['data'].encode('utf-8').lower()])[0].toarray()[0])
#         test_Y.append(dic['label'])

'''
FastText
'''
# load training data from json file and trained vectors
def get_trainning_data():
    train_X = []
    train_Y = []
    with open('data_training.json') as train_f:
        for line in train_f.readlines():
            dic = json.loads(line.strip())
            train_Y.append(dic['label'])

    with open('appreview_training_skipgram.txt') as train_v:
        for vecs in train_v.readlines():
            train_X.append([float(each) for each in vecs.strip().split()])

    return train_X, train_Y

# load test data from json file and trained vectors
def get_test_data():
    test_X = []
    test_Y = []
    with open('data_test.json') as test_f:
        for line in test_f.readlines():
            dic = json.loads(line.strip())
            test_Y.append(dic['label'])

    with open('appreview_test_skipgram.txt') as test_v:
        for vecs in test_v.readlines():
            test_X.append([float(each) for each in vecs.strip().split()])

    return test_X, test_Y


def main():
    train_X, train_Y = get_trainning_data()
    test_X, test_Y = get_test_data()

    # # Cross Validation
    # param_grid = dict(alpha=[0.001, 0.01, 0.1, 1, 10], binarize=[0, 0.01, 0.1, 1])
    #
    # grid = GridSearchCV(BernoulliNB(), param_grid=param_grid, cv=5, scoring='f1')
    # grid.fit(np.array(train_X),np.array(train_Y))
    #
    # print("The best parameters is %s, with a f1 score of %0.2f"% (grid.best_params_, grid.best_score_))
    # #The best parameters is {'binarize': 0, 'alpha': 0.1}, with a f1 score of 0.71
    #
    # refit the classifier with the best parameters chosen by gridsearch
    # clf = BernoulliNB(alpha=0.1, binarize=0.0).fit(np.array(train_X),np.array(train_Y))


    clf = GaussianNB().fit(np.array(train_X),np.array(train_Y))

    # check the performance of classifier on test set
    print "Gaussian Naive Bayes score: ", clf.score(test_X, test_Y)

    # get predicted Y for evaluation
    predict_Y = clf.predict(test_X)
    # evaluation statistics
    print precision_recall_fscore_support(test_Y, predict_Y)

    # get the probability of predict to draw ROC plot
    predict_pro_Y = clf.predict_proba(test_X)
    fpr_nb, tpr_nb, _ = roc_curve(test_Y, predict_pro_Y[:, 1])
    roc_auc = auc(fpr_nb, tpr_nb)

    print "F1 socre is : ", f1_score(test_Y, predict_Y)
    print "Gaussian Naive Bayes AUC: ", auc(fpr_nb, tpr_nb)

    plt.figure()
    plt.plot(fpr_nb,tpr_nb, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Naive Bayes ROC curve (FastText)')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
