% labels = classifySVM(svmModel,data)
% This function uses the SVM parameterized by svmModel, to classify the
% examples provided in the data parameter. It returns a vector of class
% labels, one for each row in the data set.

function labels = classifySVM(svmModel,data)
labels = svmpredict(ones(size(data,1),1),data, svmModel)';
