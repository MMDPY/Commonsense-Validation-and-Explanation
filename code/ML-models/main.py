import re
import pandas as pd
import numpy as np
# This is our visual library
import seaborn as sns
import matplotlib.pyplot as plt

from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_punctuation, lemmatize_word , remove_whitespace, remove_stopword, stem_word

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

class CommonSenseValidation():
  def __init__(self):
    self.preprocess_functions = [to_lower, remove_punctuation, remove_stopword, stem_word]
    self.readDataTaskA(path='./data/')
    self.outputpath = './runs/'
  def readDataTaskA(self, path):
    self.df_task_a_train_data = pd.read_csv(path+'Training  Data/subtaskA_data_all.csv', index_col=0)
    self.df_task_a_train_label = pd.read_csv(path+'Training  Data/subtaskA_answers_all.csv', header=None, index_col=0)

    self.df_task_a_dev_data = pd.read_csv(path+'Dev Data/subtaskA_dev_data.csv', index_col=0)
    self.df_task_a_dev_data.reset_index(drop=True, inplace=True)

    self.df_task_a_dev_label = pd.read_csv(path+'Dev Data/subtaskA_gold_answers.csv', header=None, index_col=0)
    self.df_task_a_dev_label.reset_index(drop=True, inplace=True)

    self.df_task_a_test_data = pd.read_csv(path+'Test Data/subtaskA_test_data.csv', index_col=0)
    self.df_task_a_test_data.reset_index(drop=True, inplace=True)

    self.df_task_a_test_label = pd.read_csv(path+'Test Data/subtaskA_gold_answers.csv', header=None, index_col=0)
    self.df_task_a_test_label.reset_index(drop=True, inplace=True)

    self.df_task_a_train = pd.concat([self.df_task_a_train_data, self.df_task_a_train_label], axis=1)
    self.df_task_a_dev = pd.concat([self.df_task_a_dev_data, self.df_task_a_dev_label], axis = 1)

    self.df_task_a_test = pd.concat([self.df_task_a_test_data, self.df_task_a_test_label], axis = 1)
    self.df_task_a_test.rename({1: 'label'}, axis=1, inplace=True)

    self.df_task_a_train = pd.concat([self.df_task_a_train, self.df_task_a_dev], axis=0)
    self.df_task_a_train.reset_index(drop=True, inplace=True)
    self.df_task_a_train.rename({1: 'label'}, axis=1, inplace=True)

    
    self.df_task_a_train_processed = self.separateAndPreprocessSentence0and1(self.df_task_a_train)
    self.df_task_a_test_processed = self.separateAndPreprocessSentence0and1(self.df_task_a_test)

  def separateAndPreprocessSentence0and1(self, data):
    sentences = []
    targets = []
    for index, row in data.iterrows():
      sentence_0 = row['sent0']
      sentence_1 = row['sent1']
      if row['label'] == 0:
        targets.append(0)
        targets.append(1)
      elif row['label'] == 1:
        targets.append(1)
        targets.append(0)

      # Preprocess text
      sentences.append(self.preprocessSentence(sentence_0))
      sentences.append(self.preprocessSentence(sentence_1))
      
    train_dict = {'sentence':sentences, 'label':targets}
    return pd.DataFrame(data=train_dict)

  def preprocessSentence(self, sentence):
    return preprocess_text(sentence, self.preprocess_functions)

  def showClassFrequency(self):
    # The frequency of classes (0 and 1)
    print("Frequncy of Classes:")
    label_column = self.df_task_a_train_processed["label"].value_counts()
    plt.xlabel('classes')
    plt.ylabel('#')
    label_column.plot(kind="bar")
    plt.savefig(self.outputpath+'classFrequency.png')

  def showNumberOfSentences(self):
    print("Frequncy of Sentences:")
    sentence_column = self.df_task_a_train_processed["sentence"].value_counts()
    print(sentence_column.head())

  def showNumberOfWords(self, data):
    fig = plt.figure()
    word_cnt = {}
    for idx, row in data.iterrows():
      for word in row['sentence'].split(' '):
        word_cnt[word] = word_cnt.get(word, 0) + 1

    df_word_cnt = pd.DataFrame(data={'word': list(word_cnt.keys()), 'count': list(word_cnt.values())})
    df_word_cnt = df_word_cnt.sort_values(by=['count'], ascending=False)
    # create subplot of the different data frames
    sns.barplot(x='count',y='word',data=df_word_cnt.head(30))
    fig.savefig(self.outputpath+'wordFrequency.png')
    print(df_word_cnt.head())

  def setVectorizer(self, vectorizer_type=None):
    if vectorizer_type == None or vectorizer_type == 'tfidfVectorizer':
      self.vectorizer = TfidfVectorizer(min_df = 0.0005, 
                              max_features = 100000, 
                              tokenizer = lambda x: x.split(),
                              ngram_range = (1,4))
    elif vectorizer_type == 'countVectorizer':
      self.vectorizer = CountVectorizer()

    self.X_train = self.vectorizer.fit_transform(self.df_task_a_train_processed['sentence'])
    self.X_test = self.vectorizer.transform(self.df_task_a_test_processed['sentence'])
    self.y_train = self.df_task_a_train_processed['label'] 
    self.y_test = self.df_task_a_test_processed['label']
  
  def pairwise(self, iterable):
    a = iter(iterable)
    return zip(a, a)

  def convertToYPredicted(self, y_prob):
    y_pred = []
    for x, y in self.pairwise(y_prob):
        if x[1]>=y[1]:
            y_pred.append(0)
        else:
            y_pred.append(1)
    return y_pred
    
  def runLogisticRegression(self, c):
    LR = LogisticRegression(C=c, penalty='l2')
    LR.fit(self.X_train, self.y_train)
    y_prob = LR.predict_proba(self.X_test)
    y_pred = self.convertToYPredicted(y_prob)
    return [precision_recall_fscore_support(self.df_task_a_test_label, y_pred, average='weighted'),
            accuracy_score(self.df_task_a_test_label, y_pred)
    ]
  
  def runNaiveBayse(self):
    MNB = MultinomialNB()
    MNB.fit(self.X_train, self.y_train)
    y_prob = MNB.predict_proba(self.X_test)
    y_pred = self.convertToYPredicted(y_prob)
    return [precision_recall_fscore_support(self.df_task_a_test_label, y_pred, average='weighted'),
            accuracy_score(self.df_task_a_test_label, y_pred)
    ]
  
  def runSVM(self, kernel='linear'):
    SVM = svm.SVC(kernel=kernel, probability=True)
    SVM.fit(self.X_train, self.y_train)
    y_prob = SVM.predict_proba(self.X_test)
    y_pred = self.convertToYPredicted(y_prob)
    return [precision_recall_fscore_support(self.df_task_a_test_label, y_pred, average='weighted'),
                accuracy_score(self.df_task_a_test_label, y_pred)
        ]
  
  def __str__(self):
    print("Training Data Task A (Befor Preprocess):")
    print(self.df_task_a_train.head(10))
    print('*'*20)
    print("Training Data Task A (After Preprocess):")
    print(self.df_task_a_train_processed.head(10))
    print('*'*20)
    self.showClassFrequency()
    print('*'*20)
    self.showNumberOfSentences()
    print('*'*20)
    print('\n')
    self.showNumberOfWords(self.df_task_a_train_processed)
    return ''

def main():
    commonSenseValidation = CommonSenseValidation()
    commonSenseValidation.setVectorizer()
    print(commonSenseValidation)

    print('Acc Logistic Regression:')
    acc_logistic_regression = commonSenseValidation.runLogisticRegression(c=0.5)
    print(acc_logistic_regression)

    print('Acc Naive Bayes:')
    acc_naive_bayes = commonSenseValidation.runNaiveBayse()
    print(acc_naive_bayes)

    print('Acc SVM:')
    acc_svm = commonSenseValidation.runSVM()
    print(acc_svm)

main()