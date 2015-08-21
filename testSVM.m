% this example function can be used to test whether the data loads
% correctly, and whether the SVM implementation works. It learns an SVM
% with C = 2 on the entire training data, and outputs the test error. This
% error should be XXX.

load faces.mat
svmModel = trainSVM(traindata,trainlabels,2)
predictedLabels = classifySVM(svmModel,testdata)
testError = sum(abs(predictedLabels-testlabels'))/length(testlabels)
