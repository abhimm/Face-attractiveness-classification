__author__ = 'root'
from svm_all_C import classify_SVM
import scipy.io
import math
import numpy

def classify_SVM_with_CV(train_data, train_labels, n, slack_list):
    no_instance_in_fold = int(len(train_labels)/n)

    cv_result = list()
    for j in range(n):
        cv_result.append([])

    no_of_fold = 0
    i = 0

    while no_of_fold < n:
        start = 0
        end = 0

        # prepare validation fold
        start = i
        if i + no_instance_in_fold < len(train_labels):
            end = i+no_instance_in_fold
        else:
            end = len(train_data)

        if no_of_fold == n-1:
            end = len(train_data)

        test_data = train_data[start:end]
        test_labels = train_labels[start:end]

        # prepare training folds
        train_data_cv = numpy.array([])
        train_labels_cv = numpy.array([])

        if start == 0:
            train_data_cv = train_data[end:]
            train_labels_cv = train_labels[end:]
        else:
            if end == len(train_labels):
                train_data_cv = train_data[:start]
                train_labels_cv = train_labels[:start]
            else:
                train_data_cv = numpy.concatenate((train_data[:start], train_data[end:]))
                train_labels_cv = numpy.concatenate((train_labels[:start], train_labels[end:]))

        test_result = classify_SVM(train_data_cv, train_labels_cv, test_data, slack_list)

        for j in range(len(slack_list)):
            error = 0.0
            for instance, result in test_result.items():
                if not int(result[j]) == test_labels[instance]:
                    error += 1
            cv_result[no_of_fold].append(error/len(test_data))

        no_of_fold += 1
        i += no_instance_in_fold

    aggr_result = list()
    for i in range(len(slack_list)):
        error = 0.0
        for result in cv_result:
            error += result[i]
        aggr_result.append(error/len(cv_result))

    return aggr_result

def main():
    """
    load dataset
    """
    data_file = scipy.io.loadmat('faces.mat')
    train_data = data_file['traindata']
    train_label = data_file['trainlabels']
    test_data = data_file['testdata']
    test_label = data_file['testlabels']

    cv_error_out = open("svm_cv_error.dat", "wb")
    slack_list = [10, 100, 1000, 10000, 50000, 100000, 500000, 1000000]
    cv_error = classify_SVM_with_CV(train_data, train_label, 10, slack_list)
    for i in range(len(slack_list)):
        print "Cross-validation error for C = %d : %f"%(slack_list[i], cv_error[i])
        cv_error_out.write("%f %f\n"%(math.log10(slack_list[i]), cv_error[i]))

    cv_error_out.close()

if __name__ == '__main__':
    main()