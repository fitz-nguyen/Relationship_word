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
        e = ['', 'other']
    print(e[1])
    if e[1] == '(e2,e1)':
        words.append([e2, e1])
    else:
        words.append([e1, e2])
    #words.append([e1,e2])
    comment = file.readline()
    blank = file.readline()
    # labels.append(label)
    labels.append(label.replace("(e1,e2)", "").replace("(e2,e1)", ""))
file.close()


filtered_sentence = nltk.FreqDist(labels)
label_list = list(filtered_sentence.keys())
model = Word2Vec.load('./train.w2v')
# wv = model['configuration']
# new_vec = model.infer_vector('configuration')
# print(new_vec)

X_train = np.zeros((7900, 400))
y_train = np.zeros(7900)


for i in range(7900):
    a = model[words[i][0]]
    b = model[words[i][1]]
    X_train[i] = np.concatenate([a, b])
    # print('x=%i' % i, X_train[i])
    for n, key in enumerate(label_list):
        if labels[i] == key:
            y_train[i] = int(n)
            break
        # print("train %i" % i)
print('ok')
X_test = np.zeros((100, 400))
y_test = np.zeros(100)


for i in range(7900, 8000):
    a = model[words[i][0]]
    b = model[words[i][1]]
    X_test[i - 7900] = np.concatenate([a, b])
    for n, key in enumerate(label_list):
        if labels[i] == key:
            y_test[i - 7900] = int(n)
            break

# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)
# print ('Accuracy', classifier.score(X_test, y_test))

# model = LogisticRegression(C = 1e5,
#         solver = "lbfgs", multi_class = "multinomial") # C is inverse of lam
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print("Accuracy %.2f %%" % (100*accuracy_score(y_test, y_pred.tolist())))
# classifier = SGDClassifier()
# classifier.fit(X_train, y_train)
# print ('Accuracy', classifier.score(X_test, y_test))
# model = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
# model = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=100)

# def myweight(distances):
#     sigma2 = .4 # we can change this number
#     return np.exp(-distances**2/sigma2)

# model = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2, weights = myweight)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print("Accuracy of 7NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))


alpha = 1e-1 # regularization parameter
clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(100))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
acc = 100*np.mean(y_pred == y_train)
print('training accuracy: %.2f %%' % acc)



