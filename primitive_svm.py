from sklearn import svm
import scipy.io


def classify_SVM(train_data, train_labels, test_data, test_labels, slack):
    classifier = svm.LinearSVC(C=slack)
    classifier.fit(train_data, train_labels.ravel())
    test_result = classifier.predict(test_data)
    test_error = 0.0
    for i in range(len(test_data)):
        if not int(test_result[i]) == test_labels[i][0]:
            test_error += 1
    print "Test error with C = %d : %f"%(slack, test_error)

def main():
    """
    load dataset
    """
    data_file = scipy.io.loadmat('faces.mat')
    train_data = data_file['traindata']
    train_label = data_file['trainlabels']
    test_data = data_file['testdata']
    test_label = data_file['testlabels']

    classify_SVM(train_data, train_label, test_data, test_label, slack=500)

if __name__ == '__main__':
    main()