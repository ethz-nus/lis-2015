import numpy as np
import sklearn.svm as sksvm
import sklearn.cross_validation as skcv
import sklearn.preprocessing as skpp
import sklearn.grid_search as skgs
import sklearn.metrics as skmet
import read
import scorer

to_scale = True

def partition_and_train(X, Y, validationX):
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)
    scaler = skpp.StandardScaler()
    if to_scale:
        Xtrain = scaler.fit_transform(Xtrain)
        Xtest = scaler.transform(Xtest)
        validationX = scaler.transform(validationX)
    Ytrain_Y = []
    Ytrain_Z = []
    Ytest_Y = []
    Ytest_Z = []
    for label in Ytrain:
        Ytrain_Y.append(label[0])
        Ytrain_Z.append(label[1])
    for label in Ytest:
        Ytest_Y.append(label[0])
        Ytest_Z.append(label[1])
    gammas = np.logspace(-1,5,7)
    Cs = np.logspace(0,4,5)
    scorefun = skmet.make_scorer(scorer.gss, greater_is_better=False)
    svc_Y = sksvm.SVC(cache_size=1024, C=10.0, gamma=0.1)
    svc_Z = sksvm.SVC(cache_size=1024, C=10.0, gamma=0.1)
    #classifier_Y = skgs.GridSearchCV(estimator=svc_Y, cv=5, scoring=scorefun, param_grid=dict(gamma=gammas, C=Cs))
    #classifier_Z = skgs.GridSearchCV(estimator=svc_Z, cv=5, scoring=scorefun, param_grid=dict(gamma=gammas, C=Cs))
    svc_Y.fit(Xtrain, Ytrain_Y)
    svc_Z.fit(Xtrain, Ytrain_Z)
    predicted_Y = svc_Y.predict(Xtest)
    predicted_Z = svc_Z.predict(Xtest)
    #classifier_Y.fit(Xtrain, Ytrain_Y)
    #print 'y done'
    #classifier_Z.fit(Xtrain, Ytrain_Z)
    #print(classifier_Y.best_estimator_)
    #print(classifier_Z.best_estimator_)
    #predicted_Y = classifier_Y.predict(Xtest)
    #predicted_Z = classifier_Z.predict(Xtest)
    predicted_labels = make_y_dict_format(predicted_Y, predicted_Z)
    actual_labels = make_y_dict_format(Ytest_Y, Ytest_Z)
    estimated_score = scorer.calculate_score(predicted_labels, actual_labels) 
    print 'CV Score =', estimated_score 
    print 'Score on Y =', svc_Y.score(Xtest, Ytest_Y)
    print 'Score on Z = ', svc_Z.score(Xtest, Ytest_Z)
    validation_Y = svc_Y.predict(validationX)
    validation_Z = svc_Z.predict(validationX)
    write_validation_result_to_file(validation_Y, validation_Z, estimated_score)

def make_x_row(x_dict):
    output = []
    output.extend(x_dict['catk'])
    output.extend(x_dict['catl'])
    output.extend(x_dict['numeric'])
    return output

def make_y_row(y_dict):
    output = []
    output.append(y_dict['Y'])
    output.append(y_dict['Z'])
    return output

def make_y_dict_format(y_labels, z_labels):
    data = []
    for i in range(len(y_labels)):
        data.append(dict(Y = y_labels[i], Z = z_labels[i]))
    return data  

def get_x_matrix(x):
    return map(make_x_row, x)

def get_y_matrix(y):
    return map(make_y_row, y)

def get_Y_labels(y):    
    return map(lambda y_dict : y_dict['Y'], y)   

def get_Z_labels(y):
    return map(lambda y_dict : y_dict['Z'], y)

def write_validation_result_to_file(y_prediction, z_prediction, estimated_score):
    result = []
    for i in range(len(y_prediction)):
        row = []
        row.append(int(y_prediction[i]))
        row.append(int(z_prediction[i]))
        result.append(row)
    np.savetxt('result_validate-%f.txt' % estimated_score, result, fmt="%d", delimiter=',')

train_x_file = 'project_data/train.csv'
train_y_file = 'project_data/train_y.csv'
validation_x_file = 'project_data/validate.csv'
test_x_file = 'project_data/test.csv'

train_x = read.read_x(train_x_file)
train_y = read.read_y(train_y_file)
validation_x = read.read_x(validation_x_file)
#test_x = read.read_x(test_x_file)

partition_and_train(get_x_matrix(train_x), get_y_matrix(train_y), get_x_matrix(validation_x))
