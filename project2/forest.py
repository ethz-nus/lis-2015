from sklearn.ensemble import *
from sklearn.multiclass import OneVsRestClassifier
import sklearn.cross_validation as skcv

def random_forest(X, Y):
	trainer = RandomForestClassifier(n_jobs=-1, n_estimators=53)
	return build_classifier(X, Y, trainer)
	
def extra_random_trees(X, Y):
	trainer = ExtraTreesClassifier(n_jobs=-1, n_estimators=500)
	return build_classifier(X, Y, trainer)

def forest_one_v_rest(X, Y):
	trainer = OneVsRestClassifier(ExtraTreesClassifier(n_jobs=-1, n_estimators=500))
	return build_classifier(X, Y, trainer)

def ada_boost(X, Y):
	#doesnt perform as well as RandomForest or ExtraTrees
	trainer = AdaBoostClassifier()
	return build_classifier(X, Y, trainer)

def gradient_boosting(X, Y):
	trainer = GradientBoostingClassifier(n_estimators=1200, max_depth=6)
	return build_classifier(X, Y, trainer)

def feature_select(X, Y):
	selector = RandomForestClassifier(n_jobs=-1, n_estimators=53)
	selector.fit(X, Y)
	selected = selector.transform(X)
	return selector, selected

def build_classifier(X, Y, trainer):
	Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)
	classifier = trainer.fit(Xtrain, Ytrain)
	Ypred = trainer.predict(Xtest)
	return classifier, Ypred, Ytest