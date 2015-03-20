from sklearn.ensemble import *
import sklearn.cross_validation as skcv

def random_forest(X, Y):
	trainer = RandomForestClassifier(n_jobs=-1, n_estimators=53)

	return build_classifier(X, Y, trainer)
	
def extra_random_trees(X, Y):
	trainer = ExtraTreesClassifier(n_jobs=-1, n_estimators=53)

	return build_classifier(X, Y, trainer)

def build_classifier(X, Y, trainer):
	Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)

	classifier = trainer.fit(Xtrain, Ytrain)
	Ypred = trainer.predict(Xtest)

	return classifier, Ypred, Ytest