import numpy as np
import matplotlib.pylab as plt
import csv
import datetime
import sklearn.cross_validation as skcv
import sklearn.linear_model as sklin
import sklearn.metrics as skmet
import sklearn.grid_search as skgs
import sklearn.svm as sksvm
import sklearn.preprocessing as skpp

def get_features(h):
    h = float(h)
    #return [h, np.exp(h), np.square(h), np.power(h,3), np.power(h,4)]    
    return [h]    

def get_categorical_features(val, num_of_categories):
    features = []
    for i in range(num_of_categories-1):
        if i == val:
            features.append(0)
        else:
            features.append(1)
    return features                

def read_data(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            rowvals = []
            t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            rowvals.extend(get_features(t.hour))
            #rowvals.extend(get_features(t.minute))
            #rowvals.extend(get_features(t.second))
            #rowvals.extend(get_categorical_features(t.month-1,12))
            #rowvals.extend(get_categorical_features(t.year-2013,2))
            #rowvals.extend(get_features(t.day))
            #A = float(row[1])
            #rowvals.extend(get_features(A))
            #B = int(row[2])
            #rowvals.extend(get_categorical_features(B,4))
            C = float(row[3])
            rowvals.extend(get_features(C))
            #D = float(row[4])
            #rowvals.extend(get_features(D))
            E = float(row[5])
            rowvals.extend(get_features(E))
            #F = float(row[6])
            #rowvals.extend(get_features(F))
            X.append(rowvals)
    return np.atleast_2d(X) # Change from (n,) to (n,1)

def logscore(gtruth, pred):
    pred = np.clip(pred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + pred)
    return np.sqrt(np.mean(np.square(logdif)))

X = read_data('project-1-data/train.csv')
Y = np.genfromtxt('project-1-data/train_y.csv', delimiter=',')

Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)

Xtrain_scaled = skpp.scale(Xtrain)
Xtest_scaled = skpp.scale(Xtest)

#plt.plot(Xtrain[:,0], Ytrain, 'bo')
#plt.xlim([-0.5,23.5])
#plt.ylim([0,1000])
#plt.show()
"""
regressor = sklin.Ridge()
regressor.fit(Xtrain_scaled, Ytrain)
print regressor.coef_
"""

"""
Hplot = range(25)
Xplot = np.atleast_2d([get_features(x) for x in Hplot])
Yplot = regressor.predict(Xplot)
plt.plot(Hplot, Yplot, 'r', linewidth=3)
plt.show()
"""

"""
Ypred = regressor.predict(Xtest_scaled)
print 'score =', logscore(Ytest, Ypred)
"""
"""
## Cross validation
scorefun = skmet.make_scorer(logscore)
scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
print sum(scores)/len(scores)

## How to submit
Xval = read_data('project-1-data/validate.csv')
#Ypred = regressor.predict(Xval)
#np.savetxt('result_validate.txt', Ypred)
"""
"""
##With Hyper parameters
regressor_ridge = sklin.Ridge()
param_grid = {'alpha': np.linspace(0, 100, 10)}
neg_scorefun = skmet.make_scorer(lambda x, y: -logscore(x,y))
grid_search = skgs.GridSearchCV(regressor_ridge, param_grid, scoring=neg_scorefun, cv=5)
grid_search.fit(Xtrain_scaled,Ytrain)
new_regressor = grid_search.best_estimator_
print 'grid score = ', -grid_search.best_score_
#Ypred = new_regressor.predict(Xval)
#np.savetxt('grid_validate.txt', Ypred)
"""

##Kernelized
svr_rbf = sksvm.SVR(kernel='rbf')
Xtrain_scaled = skpp.scale(Xtrain)
Xtest_scaled = skpp.scale(Xtest)
rbf_regressor = svr_rbf.fit(Xtrain, Ytrain)
y_rbf = rbf_regressor.predict(Xtest)
print 'rbf score =', logscore(Ytest, y_rbf)

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
neg_scorefun = skmet.make_scorer(lambda x,y: -logscore(x,y))
grid_search = skgs.GridSearchCV(sksvm.SVR(kernel='rbf'), param_grid, scoring=neg_scorefun)
grid_search.fit(Xtrain, Ytrain)
new_kernelized = grid_search.best_estimator_
print 'grid score = ', -grid_search.best_score_

"""
Xval = read_data('project-1-data/validate.csv')
Xval_scaled = skpp.scale(Xval)
Yd = rbf_regressor.predict(Xval_scaled)
np.savetxt('result_validate.txt', Yd)
svr_lin = sksvm.SVR(kernel='linear')
lin_regressor = svr_lin.fit(Xtrain, Ytrain)
y_lin = lin_regressor.predict(Xtest)
print 'lin score =', logscore(Ytest, y_lin)

svr_poly = sksvm.SVR(kernel='poly', degree=4)
Xtrain_scaled = skpp.scale(Xtrain)
Xtest_scaled = skpp.scale(Xtest)
poly_regressor = svr_poly.fit(Xtrain_scaled, Ytrain)
y_poly = poly_regressor.predict(Xtest_scaled)
print 'poly score =', logscore(Ytest, y_poly)
"""
