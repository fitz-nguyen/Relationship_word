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
labels = []
words = []
print("loading data ...")


while(1):
    data = file.readline()
    if data == "":
        break
    e1 = re.search(r'<e1>(.*)</e1>', data).group(1)
    e2 = re.search(r'<e2>(.*)</e2>', data).group(1)
    label = file.readline().replace("\n", "")
    # try:
    #     e = re.search(r'(.*)(\(.*\))', label).groups(2)
    # except AttributeError:
    #     e = ['', 'other']
    # if e[1] == '(e2,e1)':
    #     words.append([e2, e1])
    # else:
    #     words.append([e1, e2])
    words.append([e1,e2])
    comment = file.readline()
    blank = file.readline()
    # labels.append(label)
    labels.append(label)
file.close()
print("loaded data ... ")


filtered_sentence = nltk.FreqDist(labels)
label_list = list(filtered_sentence.keys())
model = Word2Vec.load('./train2.w2v')

X_train = np.zeros((8000, 2400))
y_train = np.zeros(8000)
for i in range(8000):
    similar = np.concatenate([model.wv.most_similar(words[i][0], topn = 5), model.wv.most_similar(words[i][1], topn = 5)])
    a = model[words[i][0]]
    b = model[words[i][1]]

    features = np.concatenate([a, b])
    for j in range(10):
        features = np.concatenate([features, model[similar[j][0]]])
    # print('x=%i' % i, X_train[i])
    X_train[i] = features
    for n, key in enumerate(label_list):
        if labels[i] == key:
            y_train[i] = int(n)
            break
        # print("train %i" % i)
print("loaded data ... ")
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=500)
print('training...')
alpha = 1e-1 # regularization parameter
clf = MLPClassifier(activation='tanh', solver='sgd', alpha=alpha, hidden_layer_sizes=(200))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = 100*np.mean(y_pred == y_test)
print('training accuracy: %.2f %%' % acc)
