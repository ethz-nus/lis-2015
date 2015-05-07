from sklearn.ensemble import *
from sklearn.decomposition import *
from sklearn.feature_selection import *
from sklearn.svm import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import sklearn.cross_validation as skcv
import sklearn.preprocessing as skpp
from sklearn.naive_bayes import *
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.neighbors import *
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.tree import *
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

def build_classifier(X, Y, trainer): 
    steps = []
    if normalise:
        normaliser = skpp.StandardScaler()
        steps.append(('normaliser', normaliser))
    if select:
        #selector = VarianceThreshold(threshold=0.05) 
        #selector = PCA(n_components="mle")
        #selector = LinearSVC(penalty="l1", dual=False)
        #selector = RandomForestClassifier(n_jobs=-1, n_estimators=300)
        #selector = RFECV(estimator=LinearSVC(penalty="l1", dual=False))
        selector = LDA(n_components=9)
        print type(selector)
        steps.append(('selector', selector))

    steps.append(('classification', trainer))
    trainer = Pipeline(steps)

    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)
    classifier = trainer.fit(Xtrain, Ytrain)

    Ypred = trainer.predict_proba(Xtest)
    return classifier, Ypred, Ytest
