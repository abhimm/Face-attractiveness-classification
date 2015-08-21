import scipy.io
import numpy as np
import math


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

    #print train_data[0]
    print len(train_label)

def calc_cosine_distance(x, y):
    denom = math.sqrt(sum(x**2)*sum(y**2))
    distance = 1 - (np.dot(x,y)/denom)
    return round(distance, 4)

if __name__ == '__main__':
    main()





