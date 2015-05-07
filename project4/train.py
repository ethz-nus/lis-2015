from methods import *
import numpy as np
import csv
from scipy.stats.mstats import mode
from sklearn.covariance import EllipticEnvelope

doTest = True

def pred_score(truth, pred):
    # truth is the actual label
    # pred is the prob array
    score = np.sum(map(lambda x: - np.log(max(0.0001, pred[x[0]][x[1]])), enumerate(truth)))
    return 1.0/len(truth) * score

def run_prediction(testX, yclassifier):
    yProbs = yclassifier.predict_proba(testX)
    print yProbs
    return yProbs

def save_prediction(outname, pred, score):
    shape = pred.shape
    score = np.asscalar(score)
    np.savetxt(outname + '-%f.txt' % score, pred, fmt='%.4f', delimiter=',')

def run_and_save_prediction(tfile, outname, yclassifier, combinedScore):
    yRes = run_prediction(tfile, yclassifier)
    save_prediction(outname, yRes, combinedScore)

def save_mode_predictions(yResults, score, filename):
    yCombined = np.nanmean(yResults, axis=0)
    save_prediction(filename, yCombined, score)

def read_data_into_rows(filepath, datatype):
    data = []
    with open(filepath, 'r') as fin:
        rows = list(map(lambda x: map(datatype, x), csv.reader(fin, delimiter=',')))
        return rows

X = np.array(read_data_into_rows("project_data/train.csv", float))
Y = np.array(read_data_into_rows("project_data/train_y.csv", int))
validate = np.array(read_data_into_rows("project_data/validate.csv", float))
test = np.array(read_data_into_rows("project_data/test.csv", float))

print np.shape(X)
print np.shape(Y)
print np.shape(validate)
print np.shape(test)

runs = 5 
scores = []
yResults = []
incScores = 0

yTestResults = []

for i in range(runs):
    ytrainer = label_propagation
    print 'running ' + ytrainer.__name__
    print 'training'
    Y = np.ravel(Y)
    yclassifier, ypred, ytruth = ytrainer(X, Y)
    
    score = pred_score(ytruth, ypred)
    scores.append(score)
    
    print score
    threshold = 0.25
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
