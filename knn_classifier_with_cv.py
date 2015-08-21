__author__ = 'root'

import scipy.io
from k_0_100_nn_classifier import classify_knn
import numpy

# K = no of nearest neighbors, n = n-fold cross validation
def classify_knn_with_cv(train_data, train_labels, K, n):
    no_instance_in_fold = int(len(train_labels)/n)

    i = 0
    cv_result = list()
    for j in range(n):
        cv_result.append([])

    no_of_fold = 0

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


        test_result = classify_knn(train_data_cv, train_labels_cv, test_data)
        for j in range(K):
            error = 0.0
            for instance, result in test_result.items():
                if not result[j] == test_labels[instance]:
                    error += 1
            cv_result[no_of_fold].append(error/len(test_result))

        no_of_fold += 1
        i += no_instance_in_fold


    final_result = list()
    for j in range(K):
        error = 0.0
        for result in cv_result:
            error += result[j]
        final_result.append(error/len(cv_result))
    return final_result

def main():
    """
    load dataset
    """
    data_file = scipy.io.loadmat('faces.mat')
    train_data = data_file['traindata']
    train_label = data_file['trainlabels']


    no_of_folds = 10

    """
    Excute cross validation
    """
    output = open('cross_validation_error.dat', 'wb')
    k_error = list()
    k_error = classify_knn_with_cv(train_data, train_label, 100, no_of_folds)
    for no_of_nearest_neighbor in range(100):
        print "k-NN for no of nearest neighbor:%d" % (no_of_nearest_neighbor + 1)
        print "Cross Validation Error with %d folds: %.03f" % (no_of_folds, k_error[no_of_nearest_neighbor])
        output.write("%d %.03f\n" % (no_of_nearest_neighbor+1, k_error[no_of_nearest_neighbor]))
    output.close()

    print "Check cross_validation_error.dat for result!"
if __name__ == '__main__':
    main()