from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
# numpy
import numpy as np
import re
file = open("TRAIN_FILE.TXT", "r")
labels = []
words = []
while(1):
    data = file.readline()
    e1 = re.search(r'<e1>(.*)</e1>', data).group(1)
    e2 = re.search(r'<e2>(.*)</e2>', data).group(1)
    label = file.readline().replace("\n", "") 
    e = re.search(r'(.*)(\(.*\))', label).group(2)
    if e == '(e1,e2)':
        words.append([e1,e2])
    else:
        words.append([e2,e1])
    comment = file.readline()
    blank = file.readline()
    if data == "":
        break
    labels.append(label.replace('(e1, e2)', '').replace('(e2, e1)', ''))

file.close()

filtered_sentence = nltk.FreqDist(labels)
label_list = list(filtered_sentence.keys())

model = Doc2Vec.load('./train.d2v')
wv = model['configuration']
print(wv)

X_train = np.zeros((7500, 200))
y_train = np.zeros(7500)


for i in range(7500):
    X_train[i] = model.docvecs['%s' % i]
    for n, key in enumerate(label_list):
        if labels[i] == key:
            y_train[i] = int(n)
            break
        # print("train %i" % i)

X_test = np.zeros((500, 200))
y_test = np.zeros(500)


for i in range(7500, 8000):
    X_test[i - 7500] = model.docvecs['%s' % i]
    for n, key in enumerate(label_list):
        if labels[i] == key:
            y_test[i - 7500] = int(n)
            break
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
print ('Accuracy', classifier.score(X_test, y_test))
