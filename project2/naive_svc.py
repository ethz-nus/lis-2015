from sklearn.svm import *
import sklearn.cross_validation as skcv

def naive_svc_train(X, Y):
	Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)
	trainer = SVC(cache_size=1024)
	
	classifier = trainer.fit(Xtrain, Ytrain)
	Ypred = trainer.predict(Xtest)

	return classifier, Ypred, Ytest

