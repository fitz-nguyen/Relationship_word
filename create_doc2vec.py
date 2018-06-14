
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
import random

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
    data = file.readline().replace("<e1>", "").replace("</e1>", "")\
        .replace("<e2>", "").replace("</e2>", "")
    label = file.readline()
    comment = file.readline()
    blank = file.readline()
    if data == "":
        break
    training.append(word_tokenize(data))


file.close()
for i in range(8000):
    sentences.append(TaggedDocument(training[i], ['%s' % i]))


# print(sentences[0])

model = Word2Vec(min_count=1, window=10, size=200, sample=1e-4, negative=5, workers=4, sg=1)

model.build_vocab(sentences)
model.train(sentences_perm(sentences), total_examples=model.corpus_count, epochs=1000)


model.save('./train.d2v')

