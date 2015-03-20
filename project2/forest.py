from sklearn.ensemble import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import sklearn.cross_validation as skcv
import sklearn.preprocessing as skpp

normalise = True
select = False

def random_forest(X, Y):
	trainer = RandomForestClassifier(n_jobs=-1, n_estimators=53)
	return build_classifier(X, Y, trainer)
	
def extra_random_trees(X, Y):
	trainer = ExtraTreesClassifier(n_jobs=-1, n_estimators=53)
	return build_classifier(X, Y, trainer)

def forest_one_v_rest(X, Y):
	trainer = OneVsRestClassifier(ExtraTreesClassifier(n_jobs=-1, n_estimators=265))
	return build_classifier(X, Y, trainer)

def ada_boost(X, Y):
	#doesnt perform as well as RandomForest or ExtraTrees
	trainer = AdaBoostClassifier()
	return build_classifier(X, Y, trainer)

def gradient_boosting(X, Y):
	trainer = GradientBoostingClassifier(n_estimators=600, max_depth=6)
	return build_classifier(X, Y, trainer)

def build_classifier(X, Y, trainer):	
	steps = []
	if normalise:
		X = cast_X_to_floats(X)
		normaliser = skpp.StandardScaler()
		steps.append(('normaliser', normaliser))
	if select:
		selector = RandomForestClassifier(n_jobs=-1, n_estimators=106)
		steps.append(('selector', selector))

	steps.append(('classification', trainer))
	trainer = Pipeline(steps)

	Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)
	classifier = trainer.fit(Xtrain, Ytrain)

	Ypred = trainer.predict(Xtest)
	return classifier, Ypred, Ytest

def cast_X_to_floats(X):
	return list(map(lambda x: map(float, x), X))