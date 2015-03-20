from read import *
from forest import *
from naive_svc import *
import numpy as np

normalise = True

def pred_score(trueYs, trueZs, predYs, predZs):
	yscore = pred_score_single(trueYs, predYs)
	print yscore
	zscore = pred_score_single(trueZs, predZs)
	print zscore
	return yscore + zscore

def pred_score_single(truth, pred):
	score = sum(map(lambda x: x[1] != pred[x[0]], enumerate(truth)))
	return 1/(2.0 * len(truth)) * score

def separate_classification_data(Y):
	Yy = list(map(lambda x: x[0], Y))
	Yz = list(map(lambda x: x[1], Y))
	return Yy, Yz

def run_and_save_prediction(testfile, outname, yclassifier, zclassifier, combinedScore):
	testX = read_data_into_rows(testfile)
	if normalise:
		testX = cast_X_to_floats(testX)
	yRes = yclassifier.predict(testX)
	zRes = zclassifier.predict(testX)

	allRes = zip(yRes, zRes)

	np.savetxt(outname + '-%f.txt' % combinedScore, allRes, fmt="%d", delimiter=',')

X = read_data_into_rows("project_data/train.csv")
Y = read_data_into_rows("project_data/train_y.csv")

Yy, Yz = separate_classification_data(Y)

ytrainer = forest_one_v_rest
ztrainer = forest_one_v_rest

yclassifier, ypred, ytruth = ytrainer(X, Yy)
zclassifier, zpred, ztruth = ztrainer(X, Yz)

score = pred_score(ytruth, ztruth, ypred, zpred)

print score

run_and_save_prediction("project_data/validate.csv", "validate", yclassifier, zclassifier, score)