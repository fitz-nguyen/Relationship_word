X_train = np.zeros((7500, 400))
y_train = np.zeros(7500)


for i in range(7500):
    X_train[i] = model.docvecs['%s' % i]
    for n, key in enumerate(label_list):
        if labels[i] == key:
            y_train[i] = int(n)
            break
    print("train %i" % i)

X_test = np.zeros((500, 400))
y_test = np.zeros(500)


for i in range(7500, 8000):
    X_test[i - 7500] = model.docvecs['%s' % i]
    for n, key in enumerate(label_list):
        if labels[i] == key:
            y_test[i - 7500] = int(n)
            break

classifier = SklearnClassifier(LogisticRegression())
classifier.train(X_train, y_train)
print ('Accuracy', nltk.classify.accuracy(classifier, X_test, y_test))