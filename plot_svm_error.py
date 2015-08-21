__author__ = 'root'

import matplotlib.pyplot as pyplot

train_file = open("./svm_train.out", "rb")
test_file = open("./svm_test.out", "rb")
cv_file = open("svm_cv_error.dat", "rb")

train_data = train_file.readlines()
test_data = test_file.readlines()
cv_data = cv_file.readlines()

plotX_train = list()
plotX_test = list()
plot_test = list()
plot_train = list()
plotX_cv = list()
plot_cv = list()

for record in train_data:
    temp = record.strip().split()
    plotX_train.append(float(temp[0]))
    plot_train.append(float(temp[1]))


for record in test_data:
    temp = record.strip().split()
    plotX_test.append(float(temp[0]))
    plot_test.append(float(temp[1]))

for record in cv_data:
    temp = record.strip().split()
    plotX_cv.append(float(temp[0]))
    plot_cv.append(float(temp[1]))

pyplot.axis([0, 7, 0.2, 0.7])
pyplot.xticks([0.5*k for k in range(13)])

pyplot.title("Test, Training and Cross-validation error for SVM")
pyplot.xlabel("Slack - log(C)[base 10]")
pyplot.ylabel("Error")
pyplot.plot(plotX_train, plot_train, label='Train Error')
pyplot.legend()
pyplot.plot(plotX_test, plot_test, label='Test Error')
pyplot.legend()
pyplot.plot(plotX_cv, plot_cv, label='Cross-validation Error')
pyplot.legend()

pyplot.savefig("svm_error_graph.png")
