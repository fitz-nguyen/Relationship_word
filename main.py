# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
from sklearn.linear_model import LogisticRegression
import nltk
import random
# numpy
import numpy as np

training = []
words = []
labels = []
testing_set = []
sentences = []
file = open("TRAIN_FILE.TXT", "r")
def sentences_perm(sentences):
    shuffled = list(sentences)
    random.shuffle(shuffled)
    return shuffled
while(1):
    data = file.readline()
    label = file.readline().replace("(e2,e1)","") \
    .replace("(e1,e2)","") \
    .replace("\n","")
    comment = file.readline()
    blank = file.readline()
    if data == "":
        break
    labels.append(label)
    training.append((word_tokenize(data), label))
    words.extend(word_tokenize(data))
file.close()

for i in range(8000):
    sentences.append(TaggedDocument(training[i][0], ['%s' % i]))


# print(sentences[0])

model = Doc2Vec(min_count=1, window=10, vector_size=300, sample=1e-4, negative=5, workers=4)

model.build_vocab(sentences)
model.train(sentences_perm(sentences), total_examples=model.corpus_count, epochs=500)


filtered_sentence = nltk.FreqDist(labels)
label_list = list(filtered_sentence.keys())
# model.save('./train.d2v')

X_train = np.zeros((7000, 300))
y_train = np.zeros(7000)

print(y_train)
for i in range(7000):
    X_train[i] = model.docvecs['%s' % i]
    for n, key in enumerate(label_list):
        if labels[i] == key:
            y_train[i] = int(n)
            break
print(y_train)
X_test = np.zeros((1000, 300))
y_test = np.zeros(1000)


for i in range(7000, 8000):
    X_test[i - 7000] = model.docvecs['%s' % i]
    for n, key in enumerate(label_list):
        if labels[i] == key:
            y_test[i - 7000] = int(n)
            break

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
print ('Accuracy', classifier.score(X_test, y_test))
