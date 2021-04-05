import glob
import numpy as np
from matplotlib import image
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix


# Function designed to read the images from a file given by <path> and return them as a numpy array
def readImages(path):
    images = []
    for eachImage in glob.glob(path):
        images.append(image.imread(eachImage))
    images = np.array(images)
    return images


def convertImagesTo2D(images):
    # Because the array is 3D, I need to reshape it into a 2D one in order for the classification to work.
    nr_of_images, x_axis, y_axis = images.shape
    images = images.reshape((nr_of_images, x_axis * y_axis))
    return images


# Function designed to read the labels from a file where, on each line, the first 10 characters represent
# the number of the image and the 11-th character is it's classification (0..9).
def readLabels(path):
    labels = []
    file = open(path)
    for eachLine in file:
        labels.append(eachLine[11])  # Because the label is always on the 11-th space.
    labels = np.array(labels)
    labels = labels.astype(np.int8)
    file.close()
    return labels


def checkPredictions(predict, labels):
    nr = 0
    for i in range(len(predict)):
        if predict[i] != labels[i]:
            nr += 1
    return (len(predict) - nr) / len(predict)


def writePredFile(pre):
    w = open("submission.txt", "w")
    w.write("id,label\n")
    for i in range(35001, 40001):
        linie = "0" + str(i) + ".png," + str(pre[i - 35001]) + "\n"
        w.write(linie)
    w.close()


# This will be the images and the labels with witch I will train my classifier.
train_images = readImages("./train/*.png")
train_images = convertImagesTo2D(train_images)
train_labels = readLabels("./train.txt")
# This is the first set of data where I will check how well does my classifier work after training.
validation_images = readImages("./validation/*.png")
validation_images = convertImagesTo2D(validation_images)
validation_labels = readLabels("./validation.txt")
# This is the testing set
test_images = readImages("./test/*.png")
test_images = convertImagesTo2D(test_images)


# ---------MULTINOMIAL NAIVE-BAYES-------------------------
'''
from sklearn.naive_bayes import MultinomialNB as MultNB

def put_values_in_bins(matrix, nr_bins):
    ret = np.digitize(matrix, nr_bins)
    return ret


best_bin = 1
max_acc = 0.0
for i in range(1, 251):
    num_bins = i
    # Stop is equal to 1 because all pixels in the images have values in the interval [0,1], so 1 is the maxim
    # value of a pixel and 0 is the minimum one
    bins = np.linspace(start=0, stop=1, num=num_bins)

    train_images_dig = put_values_in_bins(train_images, bins)
    validation_images_dig = put_values_in_bins(validation_images, bins)

    model_MNB = MultNB()
    model_MNB.fit(train_images, train_labels)
    predictions = model_MNB.predict(validation_images)
    print(i, " ", accuracy_score(validation_labels, predictions))
    if max_acc < accuracy_score(validation_labels, predictions):
        max_acc = accuracy_score(validation_labels, predictions)
        best_bin = num_bins

print("\n Maximum accuracy was recorded at ", best_bin ," number of bins. Accuracy of ", max_acc, " .")
'''
# ----------------------------------------------------------


# ----------------KNN Classifier----------------------------
'''
from sklearn.neighbors import KNeighborsClassifier
for j in range(1, 3):
    for i in range(1, 30):
        KNN = KNeighborsClassifier(n_neighbors=i, p=j)
        KNN.fit(train_images, train_labels)
        print("Pentru ", i, "vecini avem rezultatul: ", KNN.score(validation_images, validation_labels))
'''
# ----------------------------------------------------------

# ------------------SVM---------------------------
'''

from sklearn import svm

# kernels = ['rbf'] , 'poly', 'sigmoid', 'precomputed']
# cAux = [0.01, 0.5, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
# cAux = []
params = {'C': [0.1, 1, 4.44, 4.45, 4.46, 4.47, 10, 100, 1000],
          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
          'kernel': ['rbf']}
          
print("\n+++++++++++++++++++++++++++++++++++++++++++++++++")
print("+++++++++++++KERNEL ", 'rbf', "+++++++++++++++++++++")
SVM_model = svm.SVC(C=4.46, kernel='rbf')
SVM_model.fit(train_images, train_labels)
predictions = SVM_model.predict(validation_images)
print(classification_report(validation_labels, predictions))
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
'''
# ---------------------------GradientBoostingClassifier-----------------------------------------
'''
from sklearn.ensemble import GradientBoostingClassifier

# max_depth = 6
learnings = ['sqrt', 'auto']
# n-estimators = 200 --> prediction: 0.6848
# max_features = 3 --> 0.7034
# learnings = [10, 20] --> 0.74
for learning_rate in learnings:
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.58, max_features=learning_rate, max_depth=6, random_state = 0, verbose = 2)
    model.fit(train_images, train_labels)
    predictions = model.predict(validation_images)
    print("Max_Features: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(model.score(train_images, train_labels)))
    print("Accuracy score (validation): {0:.3f}".format(model.score(validation_images, validation_labels)))
    print("Predictions: ", accuracy_score(predictions, validation_labels))
    print(classification_report(validation_labels, predictions))
    print("==============================================================================================")
'''
# ------------------------------MLPClassifier--------------------------------------
'''
from sklearn.neural_network import MLPClassifier

params = {
    'hidden_layer_sizes':[(100, ), (50, 50), (30, 30, 30), (100, 100), (100, 100, 100)],
    'activation': ['relu'],
    'solver': ['sgd'], # 'adam', 'lbfgs'],
    'max_iter': [100, 150, 200],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'adaptive'],
    'n_iter_no_change': [100]
}

cv_search = GridSearchCV(mlp_classifier_model, param_grid=params, refit=True, cv=2, n_jobs=-1, verbose=1)

cv_search.fit(train_images, train_labels)

mlp_classifier_model = MLPClassifier(alpha=0.00001 ,solver="sgd", max_iter=100, momentum=0.98)
mlp_classifier_model.fit(train_images, train_labels)
print('\n\n-----Rezultate pe setul de validare --------------------------\n\n')
predictions = mlp_classifier_model.predict(validation_images)
print(classification_report(validation_labels, predictions))

prediction_test = mlp_classifier_model.predict(test_images)
writePredFile(prediction_test)
'''
