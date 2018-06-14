
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
import random
import re

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
    if data == "":
        break
    e1 = re.search(r'<e1>(.*)</e1>', data).group(1)
    e2 = re.search(r'<e2>(.*)</e2>', data).group(1)
    try:
        e = re.search(r'(.*)(\(.*\))', label).groups(2)
    except AttributeError:
        e = 0
    if e == '(e2,e1)':
        words.append([e2, e1])
    else:
        words.append([e1, e2])
    label = file.readline()
    comment = file.readline()
    blank = file.readline()
    split_word = word_tokenize(data.replace("<e1>", "").replace("</e1>", "")\
        .replace("<e2>", "").replace("</e2>", ""))
    if len(e1.split()) > 1 and len(e2.split()) == 1:
        split = np.delete(split_word, [split_word.index(e1.split()[0]), split_word.index(e1.split()[1])] )
        split = np.insert(split, split_word.index(e1.split()[0]), e1)
        training.append(split)
    elif len(e2.split()) > 1 and len(e1.split()) == 1:
        split = np.delete(split_word, [split_word.index(e2.split()[0]), split_word.index(e1.split()[1])] )
        split = np.insert(split, split_word.index(e2.split()[0]), e2)
        training.append(split)
    elif len(e2.split()) > 1 and len(e1.split()) > 1:
        split = np.delete(split_word, [split_word.index(e2.split()[0]), split_word.index(e1.split()[1])] )
        split = np.insert(split, split_word.index(e2.split()[0]), e2)
        split = np.delete(split_word, [split_word.index(e1.split()[0]), split_word.index(e1.split()[1])] )
        split = np.insert(split, split_word.index(e1.split()[0]), e1)
        training.append(split)
    else :
        training.append(split_word)



file.close()
for i in range(8000):
    sentences.append(training[i])


# print(sentences[0])


model = Word2Vec(min_count=1, window=10, size=200, sample=1e-4, negative=5, workers=4)


model.build_vocab(sentences)
model.train(sentences_perm(sentences), total_examples=model.corpus_count, epochs=1000)


model.save('./train.w2v')

