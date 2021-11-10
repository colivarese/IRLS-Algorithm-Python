import math
import numpy as np
import pandas as pd

def IRLS(X, labels, iters:int):
    D = X.shape[1]
    W = np.array([0]*X.shape[1], dtype='float64')
    y_bar = np.mean(labels) - 0.2
    w_o = math.log(y_bar/(1-y_bar)) # our intitial estimate of the coefficients
    for _ in range(iters):
        eta_i = w_o + np.dot(X, W)
        mu_i = 1/(1+np.exp(-eta_i))
        s_i = mu_i * (1 - mu_i)
        z_i = eta_i + ((labels - mu_i) / s_i)
        S = np.diag(s_i)
        tmp = np.linalg.inv(np.dot(X.T, np.dot(S, X))) 
        tmp2 = np.dot(X.T, (np.dot(S,z_i)))
        W = np.dot(tmp, tmp2)
    return W


def loadData():

    path = './dataset/'

    for _ in range(1):
        # Load training data
        trainData = pd.read_csv(path + 'trainData1.csv').to_numpy()
        tmp = pd.read_csv(path + 'trainData1.csv').columns
        tmp = [float(i) for i in tmp]
        trainData = np.insert(trainData, 0, tmp, axis=0)

        trainLabels = pd.read_csv(path + 'trainLabels1.csv').to_numpy()
        tmp = pd.read_csv(path + 'trainLabels1.csv').columns
        tmp = [float(i) for i in tmp]
        trainLabels = np.insert(trainLabels, 0, tmp, axis=0)
        trainLabels = trainLabels[:, 0]

        for i in range(2,11):
            tmp_data = pd.read_csv(path + f'trainData{i}.csv').to_numpy()
            tmp = pd.read_csv(path + f'trainData{i}.csv').columns
            tmp = [float(i) for i in tmp]
            tmp_data = np.insert(tmp_data, 0, tmp, axis=0)
            trainData = np.concatenate((trainData,tmp_data))

            tmp_labels = pd.read_csv(path + f'trainLabels{i}.csv').to_numpy()
            tmp = pd.read_csv(path + f'trainLabels{i}.csv').columns
            tmp = [float(i) for i in tmp]
            tmp_labels = np.insert(tmp_labels, 0, tmp, axis=0)
            tmp_labels = tmp_labels[:, 0]
            trainLabels = np.concatenate((trainLabels, tmp_labels))

        testData = pd.read_csv(path + 'testData.csv').to_numpy()
        tmp = pd.read_csv(path + 'testData.csv').columns
        tmp = [float(i) for i in tmp]
        testData = np.insert(testData, 0, tmp, axis=0)

        testLabels = pd.read_csv(path + 'testLabels.csv').to_numpy()
        tmp = pd.read_csv(path + 'testLabels.csv').columns
        tmp = [float(i) for i in tmp]
        testLabels = np.insert(testLabels, 0, tmp, axis=0)

        #trainData = np.c_[ np.ones(trainData.shape[0]) , trainData] 
        #testData = np.c_[ np.ones(testData.shape[0]) , testData]

        for i, n in enumerate(trainLabels):
            if n == 5:
                trainLabels[i] = 0
            else:
                trainLabels[i] = 1

    return trainData, trainLabels, testData, testLabels


def predict(W, data):
    pred = data.dot(W)
    odds = np.exp(pred)
    pred = odds / (1 + odds)
    
    if pred >= 0.5:
        return 6
    else:
        return 5

def testResult(W, X, y):
    s = 0
    for i, data in enumerate(X):
        pred = predict(W, data)
        if pred ==  y[i]:
            s += 1
    error = s / len(X)
    print(f"The error of the algorithm is: {error}")