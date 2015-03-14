"""
Description - Regression based on Radial based function Kernel
Feature parameters chosen -
    hour, weekday, month, year, C, E
    chosen based on greedy addition
Things tried which seemed to show worse results:
    Scaling to zero mean and unit variance using scikit learn preprocessing
        (showed worse results by 0.03 - 0.04 for all methods)
    Linear Regressions - Ridge, Lasso, ElasticNet
        (showed results ~1.1)
    Linear and Poly Kernel
        (showed results ~0.9 - 1.0)
    Using categorical features for weekday, hour, month, year
        (showed similar results but slower runtime due to more features)
Time to run ~ 2 minutes
"""

import numpy as np
import matplotlib.pylab as plt
import csv
import datetime
import sklearn.cross_validation as skcv
import sklearn.linear_model as sklin
import sklearn.metrics as skmet
import sklearn.grid_search as skgs
import sklearn.svm as sksvm
from sklearn.grid_search import GridSearchCV

def read_data(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            rowvals = []
            t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            rowvals.append(float(t.hour))
            rowvals.append(float(t.weekday()))
            #rowvals.append(t.minute)
            #rowvals.append(t.second)
            rowvals.append(float(t.month))
            rowvals.append(float(t.year-2012))
            #rowvals.append(t.day)
            #A = float(row[1])
            #rowvals.append(A)
            #B = float(row[2])
            #rowvals.append(B)
            C = float(row[3])
            rowvals.append(C)
            #D = float(row[4])
            #rowvals.append(D)
            E = float(row[5])
            rowvals.append(E)
            #F = float(row[6])
            #rowvals.append(F)
            X.append(rowvals)
    return np.atleast_2d(X) # Change from (n,) to (n,1)

def logscore(gtruth, pred):
    pred = np.clip(pred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + np.maximum(0, pred))
    return np.sqrt(np.mean(np.square(logdif)))

X = read_data('project-1-data/train.csv')
Y = np.genfromtxt('project-1-data/train_y.csv', delimiter=',')

Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)

"""
Radial based function kernel regressor on train data
Choice of parameters C and gamma based on:
    Grid search about logspace to determine order of magnitude
    Greedy increase and decrease to finetune
"""
avgYTrain = np.mean(Ytrain)
YTrainsd = np.std(Ytrain)
cval = max(abs(avgYTrain + 3 * YTrainsd), abs(avgYTrain - 3*YTrainsd))


svr_rbf = sksvm.SVR(kernel='rbf', cache_size=1024, C=cval, gamma=0.15, epsilon=1.75)

rbf_regressor = svr_rbf.fit(Xtrain, Ytrain)
y_rbf = rbf_regressor.predict(Xtest)
score = logscore(Ytest, y_rbf)

print ('rbf score = %f' % score)
print rbf_regressor.get_params()

"""
Read validation data and output prediction to file
Change validate.csv to test.csv to output predictions for test
"""
Xval = read_data('project-1-data/validate.csv')
Yd = rbf_regressor.predict(Xval)
np.savetxt('result_validate.txt-%f' % score, Yd)
