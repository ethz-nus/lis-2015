from read import *
from forest import *
from naive_svc import *
import numpy as np
import sklearn.preprocessing as skpp

normalise = True

def normalise_data(X, normaliser=None):
	X = list(map(lambda x: map(float, x), X))
	if normaliser == None:
		normaliser = skpp.StandardScaler()
		newX = normaliser.fit_transform(X)
	else:
		newX = normaliser.transform(X)
	return normaliser, newX


def pred_score(trueYs, trueZs, predYs, predZs):
	yscore = pred_score_single(trueYs, predYs)
	zscore = pred_score_single(trueZs, predZs)
	print "yscore "
	print yscore
	print "zscore "
	print zscore
	return yscore + zscore

def pred_score_single(truth, pred):
	score = sum(map(lambda x: x[1] != pred[x[0]], enumerate(truth)))
	return 1/(2.0 * len(truth)) * score

def separate_classification_data(Y):
	Yy = list(map(lambda x: x[0], Y))
	Yz = list(map(lambda x: x[1], Y))
	return Yy, Yz

def run_and_save_prediction(testfile, outname, yclassifier, zclassifier, combinedScore, normaliser=None):
	testX = read_data_into_rows(testfile)
	if normaliser != None:
		normaliser, testX = normalise_data(testX, normaliser)
	yRes = yclassifier.predict(testX)
	zRes = zclassifier.predict(testX)

	allRes = zip(yRes, zRes)

	np.savetxt(outname + '-%f.txt' % combinedScore, allRes, fmt="%d", delimiter=',')

X = read_data_into_rows("project_data/train.csv")
Y = read_data_into_rows("project_data/train_y.csv")

if normalise:
	normaliser, X = normalise_data(X)
else:
	normaliser = None


Yy, Yz = separate_classification_data(Y)

ytrainer = forest_one_v_rest_y
ztrainer = forest_one_v_rest_z

yclassifier, ypred, ytruth = ytrainer(X, Yy)
zclassifier, zpred, ztruth = ztrainer(X, Yz)

score = pred_score(ytruth, ztruth, ypred, zpred)

print score

run_and_save_prediction("project_data/validate.csv", "validate", yclassifier, zclassifier, score, normaliser)
