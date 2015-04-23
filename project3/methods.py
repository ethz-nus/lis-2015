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
from nolearn.dbn import DBN

normalise = True
select = False

#Trees are very slow, but naive baynes seems hopeless
def random_forest(X, Y):
    trainer = RandomForestClassifier(n_jobs=-1, n_estimators=300, max_features=None)
    return build_classifier(X, Y, trainer)
    
def extra_random_trees(X, Y):
    trainer = ExtraTreesClassifier(n_jobs=-1, n_estimators=300, max_features=None)
    return build_classifier(X, Y, trainer)

def forest_one_v_rest(X, Y):
    trainer = OneVsRestClassifier(ExtraTreesClassifier(n_jobs=-1, n_estimators=1024,  max_features=None))
    return build_classifier(X, Y, trainer)

def ada_boost(X, Y):
    #doesnt perform as well as RandomForest or ExtraTrees
    trainer = AdaBoostClassifier()
    return build_classifier(X, Y, trainer)

def gradient_boosting(X, Y):
    trainer = GradientBoostingClassifier(n_estimators=700, max_depth=6, max_features=None)
    return build_classifier(X, Y, trainer)

def naive_bayes(X, Y):
    #trainer = BernoulliNB()
    trainer = GaussianNB()
    return build_classifier(X, Y, trainer)

def nearest_centroid(X, Y):
    trainer = NearestCentroid(metric='euclidean')
    return build_classifier(X, Y, trainer)

def deep_belief_network(X, Y):
    trainer = DBN([-1, 20, 10, -1], learn_rates=0.01, epochs=5, verbose=1)
    return build_classifier(X, Y, trainer)

def linear_discriminant_analysis(X, Y):
    trainer = LDA()
    return build_classifier(X, Y, trainer)

def quadratic_discriminant_analysis(X, Y):
    trainer = QDA()
    return build_classifier(X, Y, trainer)

def logistic_regression(X, Y):
    trainer = LogisticRegression(C=10000)
    return build_classifier(X, Y, trainer)

def build_classifier(X, Y, trainer):    
    steps = []
    if normalise:
        normaliser = skpp.StandardScaler()
        steps.append(('normaliser', normaliser))
    if select:
        print "with selection"
        # selector = VarianceThreshold(threshold=0.05) 
        # selector = PCA(n_components="mle")
        #selector = LinearSVC(penalty="l1", dual=False)
        #selector = RandomForestClassifier(n_jobs=-1, n_estimators=300)
        #selector = RFECV(estimator=RandomForestClassifier(n_jobs=-1))
        #selector = BernoulliRBM(n_components=100, learning_rate=0.1, n_iter=10)
        selector = LDA(n_components=9)
        steps.append(('selector', selector))

    steps.append(('classification', trainer))
    trainer = Pipeline(steps)

    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)
    classifier = trainer.fit(Xtrain, Ytrain)

    Ypred = trainer.predict(Xtest)
    return classifier, Ypred, Ytest
