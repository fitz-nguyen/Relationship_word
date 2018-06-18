from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
import nltk
import numpy as np
import re


file = open("TRAIN_FILE.TXT", "r")
file_test = open("TEST_FILE.txt", "r")
labels = []
words = []
words_test = []
testing_set = []


print("loading training data in TRAIN_FILE.TXT ...")
while(1):
    data = file.readline()
    if data == "":
        break
    e1 = re.search(r'<e1>(.*)</e1>', data).group(1)
    e2 = re.search(r'<e2>(.*)</e2>', data).group(1)
    label = file.readline().replace("\n", "")
    words.append([e1,e2])
    comment = file.readline()
    blank = file.readline()
    # labels.append(label)
    labels.append(label)
file.close()
print("finished training data ...")


print("loading features for training_set ...")
filtered_sentence = nltk.FreqDist(labels)
label_list = list(filtered_sentence.keys())
model = Word2Vec.load('./train2.w2v')
X_train = np.zeros((8000, 400))
y_train = np.zeros(8000)
for i in range(8000):
    a = model[words[i][0]]
    b = model[words[i][1]]
    X_train[i] = np.concatenate([a, b])
    for n, key in enumerate(label_list):
        if labels[i] == key:
            y_train[i] = int(n)
            break
print("finished ...")


print("loading testing data in TEST_FILE.txt ...")
while(1):
    data = file_test.readline()
    if data == "":
        break
    e1 = re.search(r'<e1>(.*)</e1>', data).group(1)
    e2 = re.search(r'<e2>(.*)</e2>', data).group(1)
    words_test.append([e1, e2])
file_test.close()
print("finished testing data ... ")


model_test = Word2Vec.load('./test.w2v')
print("loading features for testing_set ...")
X_testing = np.zeros((2717, 400))
for i in range(2717):
    a = model_test[words_test[i][0]]
    b = model_test[words_test[i][1]]
    X_testing[i] = np.concatenate([a, b])
print("finished ...")


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=500)
print('training...')
alpha = 1e-1 # regularization parameter
clf = MLPClassifier(activation='tanh', solver='sgd', alpha=alpha, hidden_layer_sizes=(200))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = 100*np.mean(y_pred == y_test)
print('training accuracy: %.2f %%' % acc)


print("Save the results ... ")
results = clf.predict(X_testing)
file_results = open("results.txt", "w")
for i in range(8001, 10717):
    for n, key in enumerate(label_list):
        if results[i-8001] == n:
            file_results.write(str(i) + " " + key + "\n")
            break
print("finished.")





















