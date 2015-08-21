__author__ = 'root'

import scipy.io
from calculate_distance import calc_cosine_distance
from operator import itemgetter

def classify_knn(train_data, train_labels, test_data, K):

    # calculate cosine distance for every test instance wrt training instances
    distance = dict()

    for j in range(len(test_data)):
        distance[j] = list()
        for i in range(len(train_data)):
            distance[j].append([i, calc_cosine_distance(test_data[j], train_data[i])])

    test_labels = dict()
    positive_label = 1
    negative_label = 2

    # classify test instances based on K nearest neighbors label

    for test_instance, dist_list in distance.items():
        no_positive_label = 0
        no_negative_label = 0

        for entry in sorted(dist_list, key=itemgetter(1))[: K+1]:

            if train_labels[entry[0]] == positive_label:
                no_positive_label += 1
            else:
                no_negative_label += 1

        if no_positive_label > no_negative_label:
            test_labels[test_instance] = positive_label
        else:
            test_labels[test_instance] = negative_label

    return test_labels

def main():
    """
    load dataset
    """
    data_file = scipy.io.loadmat('faces.mat')
    train_data = data_file['traindata']
    train_label = data_file['trainlabels']
    test_data = data_file['testdata']
    test_lable = data_file['testlabels']
    eval_data = data_file['evaldata']
    print classify_knn(train_data, train_label, train_data, 8)

if __name__ == '__main__':
    main()