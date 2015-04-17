import h5py
from methods import *
import numpy as np
from scipy.stats.mstats import mode
from sklearn.covariance import EllipticEnvelope

doTest = False

def pred_score(truth, pred):
	score = np.sum(map(lambda x: x[1] != pred[x[0]], enumerate(truth)))
	return 1/(2.0 * len(truth)) * score

def run_prediction(tfile, yclassifier):
	testX = tfile['data']
	yRes = yclassifier.predict(testX)
	return yRes

def save_prediction(outname, pred, score):
	out = h5py.File(outname + '-%f.h5' % score , 'w')
	out['label'] = pred
	out.flush()

def run_and_save_prediction(tfile, outname, yclassifier, combinedScore):
	yRes = run_prediction(tfile, yclassifier)
	save_prediction(outname, yRes, combinedScore)

def save_mode_predictions(yResults, score, filename):
	yCombined = mode(np.array(yResults))[0]
	save_prediction(filename, yCombined, combinedScore)

train = h5py.File("project_data/train.h5", "r")
validate = h5py.File("project_data/validate.h5", "r")
test = h5py.File("project_data/test.h5", "r")

X = train['data']
Y = train['label']

runs = 10
scores = []
yResults = []
incScores = 0

yTestResults = []

for i in range(runs):
	print 'running'
	ytrainer = naive_bayes
	print 'training'
	Y = np.ravel(Y)
	yclassifier, ypred, ytruth = ytrainer(X, Y)
	print 'predicting'
	score = pred_score(ytruth, ypred)
	scores.append(score)

	print score
	threshold = 0.21
	if score < threshold:
		yRes = run_prediction(validate, yclassifier)
		yResults.append(yRes)
		
		incScores += score
		if doTest:
			yTestRes = run_prediction(test, yclassifier)
			yTestResults.append(yTestRes)
		
if yResults:
	combinedScore = incScores / len(yResults)
	save_mode_predictions(yResults, combinedScore, "validate")
	if doTest:
		save_mode_predictions(yTestResults, combinedScore, "test")


print np.mean(scores)
print np.std(scores)