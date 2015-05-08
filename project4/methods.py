import numpy as np
from sklearn.pipeline import Pipeline
import sklearn.cross_validation as skcv
import sklearn.preprocessing as skpp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.lda import LDA
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation

normalise = False
select = False

def label_spreading(X, Y):
    trainer = LabelSpreading(kernel='knn')
    return build_classifier(X, Y, trainer)

def label_propagation(X, Y):
    trainer = LabelPropagation(kernel='knn')
    return build_classifier(X, Y, trainer)

def split_train_set(X, Y):
    labX = []
    labY = []
    uX = []
    uY = []
    for i in range(len(Y)):
        c = Y[i]
        if c == -1:
            uX.append(X[i])
            uY.append(c)
        else:
            labX.append(X[i])
            labY.append(c)
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(labX, labY, train_size=0.75)
    Xtrain = np.append(Xtrain, uX, axis=0)
    Ytrain = np.append(Ytrain, uY, axis=0)
    return Xtrain, Xtest, Ytrain, Ytest

def build_classifier(X, Y, trainer): 
    steps = []
    if normalise:
        normaliser = skpp.StandardScaler()
        steps.append(('normaliser', normaliser))
    if select:
        selector = LDA(n_components=9)
        print type(selector)
        steps.append(('selector', selector))

    steps.append(('classification', trainer))
    trainer = Pipeline(steps)

    Xtrain, Xtest, Ytrain, Ytest = split_train_set(X, Y)
    classifier = trainer.fit(Xtrain, Ytrain)

    Ypred = trainer.predict_proba(Xtest)
    return classifier, Ypred, Ytest
