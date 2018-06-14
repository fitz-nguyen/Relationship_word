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
    if data == "":
        break
    e1 = re.search(r'<e1>(.*)</e1>', data).group(1)
    e2 = re.search(r'<e2>(.*)</e2>', data).group(1)
    label = file.readline().replace("\n", "")
    try:
        e = re.search(r'(.*)(\(.*\))', label).groups(2)
    except AttributeError:
        e = 0
    if e == '(e2,e1)':
        words.append([e2, e1])
    else:
        words.append([e1, e2])
    comment = file.readline()
    blank = file.readline()
    labels.append(label.replace('(e1,e2)', '').replace('(e2,e1)', ''))
file.close()


filtered_sentence = nltk.FreqDist(labels)
label_list = list(filtered_sentence.keys())
model = Doc2Vec.load('./train.d2v')
# wv = model['configuration']
# new_vec = model.infer_vector('configuration')
# print(new_vec)

X_train = np.zeros((7500, 400))
y_train = np.zeros(7500)


for i in range(7500):
    a = model.infer_vector(words[i][0])
    b = model.infer_vector(words[i][0])
    X_train[i] = np.concatenate([a, b])
    # print('x=%i' % i, X_train[i])
    for n, key in enumerate(label_list):
        if labels[i] == key:
            y_train[i] = int(n)
            break
        # print("train %i" % i)
print('ok')
X_test = np.zeros((500, 400))
y_test = np.zeros(500)


for i in range(7500, 8000):
    a = model.infer_vector(words[i][0])
    b = model.infer_vector(words[i][0])
    X_test[i-7500] = np.concatenate([a, b])
    for n, key in enumerate(label_list):
        if labels[i] == key:
            y_test[i - 7500] = int(n)
            break
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
print ('Accuracy', classifier.score(X_test, y_test))
