import h5py
from methods import *
import numpy as np
from scipy.stats.mstats import mode
from sklearn.covariance import EllipticEnvelope

doTest = False

def pred_score(truth, pred):
	score = np.sum(map(lambda x: x[1] != pred[x[0]], enumerate(truth)))
	return 1.0/len(truth) * score

def run_prediction(tfile, yclassifier):
	testX = np.array(tfile['data'])
	yRes = yclassifier.predict(testX)
	yProbs = yclassifier.predict_proba(testX)
	print yProbs
	
	return yRes

def save_prediction(outname, pred, score):
	# out = h5py.File(outname + '-%f.h5' % score , 'w')
	shape = pred.shape
	pred = pred.reshape((shape[1], shape[0]))
	# out.flush()
	np.savetxt(outname + '-%f.txt' % score, pred, fmt="%d", delimiter=',')

def run_and_save_prediction(tfile, outname, yclassifier, combinedScore):
	yRes = run_prediction(tfile, yclassifier)
	save_prediction(outname, yRes, combinedScore)

def save_mode_predictions(yResults, score, filename):
	yCombined = mode(np.array(yResults))[0]
	save_prediction(filename, yCombined, combinedScore)

train = h5py.File("project_data/train.h5", "r")
validate = h5py.File("project_data/validate.h5", "r")
test = h5py.File("project_data/test.h5", "r")

X = np.array(train['data'])
Y = np.array(train['label'])

runs = 5  
scores = []
yResults = []
incScores = 0

yTestResults = []

for i in range(runs):
	ytrainer = deep_belief_network
	print 'running ' + ytrainer.__name__
	print 'training'
	Y = np.ravel(Y)
	yclassifier, ypred, ytruth = ytrainer(X, Y)
	
	score = pred_score(ytruth, ypred)
	scores.append(score)
	
	print score
	threshold = 0.28
	if score < threshold:
		print 'predicting'
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
