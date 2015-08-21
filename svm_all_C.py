__author__ = 'root'
from sklearn import svm
import scipy.io
import math


def classify_SVM(train_data, train_labels, test_data, slack_list):

    aggr_test_result = dict()

    for i in range(len(test_data)):
        aggr_test_result[i] = list()

    for slack in slack_list:
        classifier = svm.LinearSVC(C=slack)
        classifier.fit(train_data, train_labels.ravel())
        test_result = classifier.predict(test_data)
        for i in range(len(test_data)):
            aggr_test_result[i].append(test_result[i])

    return aggr_test_result

def execute_SVM(train_data, train_labels, test_data, test_labels):
    slack_list = [10, 100, 1000, 10000, 50000, 100000, 500000, 1000000]
    test_result = classify_SVM(train_data, train_labels, test_data, slack_list)
    test_error = list()
    for i in range(len(slack_list)):
        error = 0.0
        for j in range(len(test_data)):
            if not int(test_result[j][i]) == test_labels[j][0]:
                error += 1
        test_error.append(error/len(test_data))

    return test_error
    #print "Test error with C = %d : %f"%(slack, test_error)

def main():
    """
    load dataset
    """
    data_file = scipy.io.loadmat('faces.mat')
    train_data = data_file['traindata']
    train_label = data_file['trainlabels']
    test_data = data_file['testdata']
    test_label = data_file['testlabels']

    slack_list = [10, 100, 1000, 10000, 50000, 100000, 500000, 1000000]

    train_error = execute_SVM(train_data, train_label, train_data, train_label)
    test_error = execute_SVM(train_data, train_label, test_data, test_label)

    train_output = open('./svm_train.out', 'wb')
    test_output = open('./svm_test.out', 'wb')

    for i in range(len(slack_list)):
        test_output.write("%f %f\n"%(math.log10(slack_list[i]), test_error[i]))
        train_output.write("%f %f\n"%(math.log10(slack_list[i]), train_error[i]))
        print "Train error for C = %d : %f"%(slack_list[i], train_error[i])
        print "Test error for C = %d : %f"%(slack_list[i], test_error[i])
    train_output.close()
    test_output.close()


if __name__ == '__main__':
    main()