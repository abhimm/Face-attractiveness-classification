__author__ = 'root'
__author__ = 'Abhinav'
import matplotlib.pyplot as pyplot

train_file = open("./train.out", "rb")
test_file = open("./test.out", "rb")
cv_file = open("cross_validation_error.dat", "rb")

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
    plotX_train.append(int(temp[0]))
    plot_train.append(float(temp[1]))


for record in test_data:
    temp = record.strip().split()
    plotX_test.append(int(temp[0]))
    plot_test.append(float(temp[1]))

for record in cv_data:
    temp = record.strip().split()
    plotX_cv.append(int(temp[0]))
    plot_cv.append(float(temp[1]))

pyplot.axis([0, 105, 0.3, 0.55])
#pyplot.yticks([0.3, 0.4, 0.5, 0.6])

pyplot.title("Test, Training and Cross-validation error for K-NN")
pyplot.xlabel("No of nearest neighbor-K")
pyplot.ylabel("Error")
pyplot.plot(plotX_train, plot_train, label='Train Error')
pyplot.legend()
pyplot.plot(plotX_test, plot_test, label='Test Error')
pyplot.legend()
pyplot.plot(plotX_cv, plot_cv, label='Cross-validation Error')
pyplot.legend()

pyplot.savefig("error_graph.png")
