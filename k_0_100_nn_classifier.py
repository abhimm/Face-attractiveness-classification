__author__ = 'root'

import scipy.io
from calculate_distance import calc_cosine_distance
from operator import itemgetter

def classify_knn(train_data, train_labels, test_data):

    # calculate cosine distance for every test instance wrt training instances
    distance = dict()

    for j in range(len(test_data)):
        distance[j] = list()
        for i in range(len(train_data)):
            distance[j].append([i, calc_cosine_distance(test_data[j], train_data[i])])

    test_labels = dict()
    positive_label = 1
    negative_label = 2
    for i in range(len(test_data)):
        test_labels[i] = list()
    # classify test instances based on K nearest neighbors label

    for test_instance, dist_list in distance.items():

        sorted_dist_list = sorted(dist_list, key=itemgetter(1))
        no_positive_label = 0
        no_negative_label = 0

        for j in range(100):

            if train_labels[sorted_dist_list[j][0]] == positive_label:
                no_positive_label += 1
            else:
                no_negative_label += 1

            if no_positive_label > no_negative_label:
                test_labels[test_instance].append(positive_label)
            else:
                test_labels[test_instance].append(negative_label)


    return test_labels

def execute_knn(train_data, train_labels, test_data, test_labels):
    test_result = classify_knn(train_data,train_labels,test_data)
    test_error = list()
    for k in range(100):
        error = 0.0
        for instance, result in test_result.items():
            if not result[k] == test_labels[instance]:
                error += 1
        test_error.append(error/len(test_result))

    return test_error

def main():
    """
    load dataset
    """
    data_file = scipy.io.loadmat('faces.mat')
    train_data = data_file['traindata']
    train_label = data_file['trainlabels']
    test_data = data_file['testdata']
    test_label = data_file['testlabels']
    eval_data = data_file['evaldata']

    test_error = execute_knn(train_data, train_label, test_data, test_label)
    train_error = execute_knn(train_data, train_label, train_data, train_label)

    train_output = open('./train.out', 'wb')
    test_output = open('./test.out', 'wb')

    for no_of_nearest_neighbor in range(100):
        print "k-NN for no of nearest neighbor:%d" % (no_of_nearest_neighbor + 1)
        print "Training error: %.03f" % train_error[no_of_nearest_neighbor]
        print "Test error: %.03f" % test_error[no_of_nearest_neighbor]
        print "--------------------------------------------"
        train_output.write("%d %.03f\n" % (no_of_nearest_neighbor+1, train_error[no_of_nearest_neighbor]))
        test_output.write("%d %.03f\n" % (no_of_nearest_neighbor+1, test_error[no_of_nearest_neighbor]))
    train_output.close()
    test_output.close()

if __name__ == '__main__':
    main()