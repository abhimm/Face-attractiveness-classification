% svmModel = trainSVM(data,labels,C)
% this function uses libsvm to learn a linear support vector machine
% data is the data to be used for training, labels is the vector of class
% labels, and C is the slack penalty > 0
% The function returns the SVM parameters, to be used with classifySVM

function svmModel = trainSVM(data,labels,C)
svmModel = svmtrain(labels, data, sprintf('-t 0 -c %f', C));
