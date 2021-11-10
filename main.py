from utils import *


trainData, trainLabels, testData, testLabels = loadData()
W = IRLS(X=trainData, labels=trainLabels, iters= 5)
testResult(W, testData, testLabels)