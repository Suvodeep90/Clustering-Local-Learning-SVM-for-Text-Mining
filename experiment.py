from __future__ import division, print_function
import pickle
import pdb
import os
import time
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn import metrics
import gensim
import random
from learners import SK_SVM
from tuner import DE_Tune_ML
from model import PaperData
from utility import study
from results import results_process
import numpy as np
import wget
import zipfile
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import pandas as pd
import sys
import collections


def tune_learner(learner, train_X, train_Y, tune_X, tune_Y, goal, target_class=None):
  if not target_class:
    target_class = goal
  clf = learner(train_X, train_Y, tune_X, tune_Y, goal)
  tuner = DE_Tune_ML(clf, clf.get_param(), goal, target_class)
  return tuner.Tune()


def load_vec(d, data, use_pkl=False, file_name=None):
  if use_pkl:
    if os.path.isfile(file_name):
      with open(file_name, "rb") as my_pickle:
        return pickle.load(my_pickle)
  else:
    # print("call get_document_vec")
    return d.get_document_vec(data, file_name)


def print_results(clfs):
  file_name = time.strftime(os.path.sep.join([".", "results",
                                              "%Y%m%d_%H:%M:%S.txt"]))
  file_name = os.path.sep.join(["20171103.txt"])
  content = ""
  for each in clfs:
    content += each.confusion
  print(content)
  with open(file_name, "w") as f:
    f.write(content)
  results_process.reports(file_name)


def get_acc(cm):
  out = []
  for i in range(4):
    out.append(cm[i][i] / 400)
    print(cm)
  return out


"""
  :param word2vec_src:str, path of word2vec model
  :param repeats:int, number of repeats
  :param fold: int,number of folds
  :param tuning: boolean, tuning or not.
  :return: None
  """


@study
def run_tuning_SVM(word2vec_src, repeats=1, fold=10, tuning=True):
  print("# word2vec:", word2vec_src)
  word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
  data = PaperData(word2vec=word2vec_model)
  train_pd = load_vec(data, data.train_data, file_name=False)
  test_pd = load_vec(data, data.test_data, file_name=False)
  learner = [SK_SVM][0]
  # learner = neighbors.KNeighborsClassifier(n_neighbors = 4)
  goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F",
          7: "Micro_F"}[6]
  F = {}
  clfs = []
  for i in range(repeats):  # repeat n times here
    kf = StratifiedKFold(train_pd.loc[:, "LinkTypeId"].values, fold,
                         shuffle=True)
    for train_index, tune_index in kf:
      train_data = train_pd.ix[train_index]
      tune_data = train_pd.ix[tune_index]
      train_X = train_data.loc[:, "Output"].values
      train_Y = train_data.loc[:, "LinkTypeId"].values
      tune_X = tune_data.loc[:, "Output"].values
      tune_Y = tune_data.loc[:, "LinkTypeId"].values
      test_X = test_pd.loc[:, "Output"].values
      test_Y = test_pd.loc[:, "LinkTypeId"].values
      print("X0")
      params, evaluation = tune_learner(learner, train_X, train_Y, tune_X,
                                        tune_Y, goal) if tuning else ({}, 0)
      print("X1")
      clf = learner(train_X, train_Y, test_X, test_Y, goal)
      print("X2")
      F = clf.learn(F, **params)
      print("X3")
      clfs.append(clf)
  print_results(clfs)


"""
  Run SVM+word embedding experiment !
  This is the baseline method.
  :return:None
  """
#@study

# returns the svm model
def run_SVM(word2vec_src, train_pd):
  print("# word2vec:", word2vec_src)
  clf = svm.SVC(kernel="rbf", gamma=0.005)
  word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
  data = PaperData(word2vec=word2vec_model)
  print("Train data: " + str(train_pd.shape))
  if train_pd is None: train_pd = load_vec(
      data, data.train_data, use_pkl=False)
  train_X = train_pd.loc[:, "Output"].tolist()
  train_Y = train_pd.loc[:, "LinkTypeId"].tolist()
  clf.fit(train_X, train_Y)
  return clf

# parses and returns a given svm in the format of dictionary -
# [class](precision, recall, f1score, support)
def results_SVM(clf, test_X, test_Y):
  predicted = clf.predict(test_X)
  # labels: ["Duplicates", "DirectLink","IndirectLink", "Isolated"]
  report_gen = metrics.classification_report(
      test_Y, predicted, labels=["1", "2", "3", "4"], digits=3)
  parsed_report = parse_classification_report(report_gen)
  return parsed_report
 #cm=metrics.confusion_matrix(test_Y, predicted, labels=["1", "2", "3", "4"])
  #print("accuracy  ", get_acc(cm))

def total_summary(result_set):
  # have access to the all the parsed svm reports
  # need to traverse through and keep tally
  t_precision =0
  t_recall = 0
  t_f1_score = 0
  t_support = 0
  
  # defaults values are none in these class totals 
  for l in range(len(result_set)):
      print(result_set[l]) 

  t_class_dic = {}
  for keys in (1, 2, 3, 4, 'avg'):
        t_class_dic[keys] = [0, 0, 0, 0]
        
  ''' for keys in (1, 2, 3, 4, 'avg'):
    for l in range(len(result_set)):
     # have access to the one svm
     for i in (1,2,3,4):
        t_class_dic[keys][i] += result_set[l][keys][i]
 '''
  for keys, values in t_class_dic.items():
    print("How am I looking")
    print(keys)
    print(values)
  
    

  
    # there has to be an efficient way to coressponding sums
  
def run_kmeans(word2vec_src):

  print("# word2vec:", word2vec_src)
  word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
  data = PaperData(word2vec=word2vec_model)
  train_pd = load_vec(data, data.train_data, use_pkl=False)
  test_pd = load_vec(data, data.test_data, use_pkl=False)
  train_X = train_pd.loc[:, "Output"].tolist()

  #numClusters = optimalK(train_X)
  numClusters = 5
  print("Found optimal k: " + str(numClusters))
  clf = KMeans(n_clusters=numClusters,
               init='k-means++', max_iter=200, n_init=1)
  clf.fit(train_X)

  svm_models = []  # maintain a list of svms
  data.train_data['clabel'] = clf.labels_
  for l in range(numClusters):
    cluster = data.train_data.loc[data.train_data['clabel'] == l]
    svm_models.append(run_SVM(word2vec_src, cluster))


  svm_results = [] # maintain a list of svm results
  test_X = test_pd.loc[:, "Output"].tolist()
  predicted = clf.predict(test_X)
  data.test_data['clabel'] = predicted
  for l in range(numClusters):
    #print("Label " + str(l))
    cluster = data.test_data.loc[data.test_data['clabel'] == l]
    svm_model = svm_models[l]
    cluster_X = cluster.loc[:, "Output"].tolist()
    cluster_Y = cluster.loc[:, "LinkTypeId"].tolist()
    svm_results.append(results_SVM(svm_model, cluster_X, cluster_Y))# store all the SVM result report in a dictionary

    # call the helper method to summarize the svm results
  total_summary(svm_results)

# Source: https://anaconda.org/milesgranger/gap-statistic/notebook


def optimalK(data, nrefs=3, maxClusters=15):
  """
  Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
  Params:
      data: ndarry of shape (n_samples, n_features)
      nrefs: number of sample reference datasets to create
      maxClusters: Maximum number of clusters to test for
  Returns: (gaps, optimalK)
  """
  gaps = np.zeros((len(range(1, maxClusters)),))
  resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
  for gap_index, k in enumerate(range(1, maxClusters)):

    # Holder for reference dispersion results
    refDisps = np.zeros(nrefs)

      # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
    for i in range(nrefs):

      # Create new random reference set
      # randomReference = np.random.random_sample(size=data.shape)

      # Fit to it
      km = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=1)
      km.fit(data)

      refDisp = km.inertia_
      refDisps[i] = refDisp

      # Fit cluster to original data and create dispersion
      km = KMeans(k)
      km.fit(data)

      origDisp = km.inertia_
      # print(str(i+1) + ": " + str(origDisp))

      # Calculate gap statistic
      gap = np.log(np.mean(refDisps)) - np.log(origDisp)

      # Assign this loop's gap statistic to gaps
      gaps[gap_index] = gap

      resultsdf = resultsdf.append(
          {'clusterCount': k, 'gap': gap}, ignore_index=True)

  # return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
  return gaps.argmax()

# Not used, but wanted to put this code somewhere


def results_kmeans(clf, train_X, train_Y, test_X, test_Y):
  predicted = clf.predict(test_X)
  print("Homogeneity: %0.3f" % metrics.homogeneity_score(train_Y, clf.labels_))
  print("Completeness: %0.3f" %
        metrics.completeness_score(train_Y, clf.labels_))
  print("V-measure: %0.3f" % metrics.v_measure_score(train_Y, clf.labels_))
  print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(train_Y, clf.labels_))
  print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(train_X, clf.labels_, sample_size=1000))



"""
Parse a sklearn classification report into a dict keyed by class name
and containing a tuple (precision, recall, fscore, support) for each class
Reference: https://gist.github.com/julienr/6b9b9a03bd8224db7b4f
"""
def parse_classification_report(clfreport):

    lines = clfreport.split('\n')
    # Remove empty lines
    lines = list(filter(lambda l: not len(l.strip()) == 0, lines))

    # Starts with a header, then score for each class and finally an average
    header = lines[0]
    cls_lines = lines[1:-1]
    avg_line = lines[-1]

    assert header.split() == ['precision', 'recall', 'f1-score', 'support']
    assert avg_line.split()[0] == 'avg'

    # class names can have spaces - figure the width of the class field
    # using indentation of the precision header
    cls_field_width = len(header) - len(header.lstrip())

    # Now, collect all the class names and score in a dict
    def parse_line(l):
        """Parse a line of classification_report"""
        cls_name = l[:cls_field_width].strip()
        precision, recall, fscore, support = l[cls_field_width:].split()
        precision = float(precision)
        recall = float(recall)
        fscore = float(fscore)
        support = int(support)
        return (cls_name, precision, recall, fscore, support)

    data = collections.OrderedDict()
    for l in cls_lines:
        ret = parse_line(l)
        cls_name = ret[0]
        scores = ret[1:]
        data[cls_name] = scores

    data['avg'] = parse_line(avg_line)[1:]  # average
    return data




def prepare_word2vec():
  print("Downloading pretrained word2vec models")
  url = "https://zenodo.org/record/807727/files/word2vecs_models.zip"
  file_name = wget.download(url)
  with zipfile.ZipFile(file_name, "r") as zip_ref:
    zip_ref.extractall()

########################## MAIN ##########################
if __name__ == "__main__":
  word_src = "word2vecs_models"
  if not os.path.exists(word_src):
    prepare_word2vec()
  elif len(os.listdir(word_src)) == 0:
    os.rmdir(word_src)
    prepare_word2vec()
  for x in range(1):
    random.seed(x)
    np.random.seed(x)
    myword2vecs = [os.path.join(word_src, i) for i in os.listdir(word_src)
                   if "syn" not in i]
    
    # print("\n########### Plain SVM ###########")
    # run_SVM(myword2vecs[x], None)
    # run_tuning_SVM(myword2vecs[x])
    print("\n########### SVM with kmeans ###########")
    kmeans = run_kmeans(myword2vecs[x])
