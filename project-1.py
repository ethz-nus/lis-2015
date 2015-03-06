import numpy as np
import matplotlib.pylab as plt
import csv
import datetime
import sklearn.cross_validation as skcv
import sklearn.linear_model as sklin
import sklearn.metrics as skmet
import sklearn.grid_search as skgs

def get_features(h):
    return [h, np.exp(h)]

def read_data(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            X.append(get_features(t.hour))
    return np.atleast_2d(X) # Change from (n,) to (n,1)

def logscore(gtruth, pred):
    pred = np.clip(pred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + pred)
    return np.sqrt(np.mean(np.square(logdif)))

X = read_data('train.csv')
Y = np.genfromtxt('train_y.csv', delimiter=',')

Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)

#plt.plot(Xtrain[:,0], Ytrain, 'bo')
#plt.xlim([-0.5,23.5])
#plt.ylim([0,1000])
#plt.show()

regressor = sklin.Ridge()
regressor.fit(Xtrain, Ytrain)
#print regressor.coef_

Hplot = range(25)
Xplot = np.atleast_2d([get_features(x) for x in Hplot])
Yplot = regressor.predict(Xplot)
#plt.plot(Hplot, Yplot, 'r', linewidth=3)
#plt.show()

Ypred = regressor.predict(Xtest)
#print 'score =', logscore(Ytest, Ypred)

## Cross validation
scorefun = skmet.make_scorer(logscore)
scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
#print sum(scores)/len(scores)

## How to submit
Xval = read_data('validate.csv')
Ypred = regressor.predict(Xval)
#np.savetxt('result_validate.txt', Ypred)

## With Hyper parameters
regressor_ridge = sklin.Ridge()
param_grid = {'alpha': [1,10,50,100]}
neg_scorefun = skmet.make_scorer(lambda x, y: -logscore(x,y))
grid_search = skgs.GridSearchCV(regressor_ridge, param_grid, scoring=neg_scorefun, cv=5)
grid_search.fit(X,Y)
new_regressor = grid_search.best_estimator_
Ypred = new_refressor.predict(Xval)
#np.savetxt('grid_validate.txt', Ypred)'
