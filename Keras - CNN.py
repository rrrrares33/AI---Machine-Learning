# For path.
import glob

# For arrays and plotting.
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt

# For classification_report.
from sklearn.metrics import classification_report, confusion_matrix

# Everything for the Convultional Neural Network.
import keras
from tensorflow.keras import datasets, layers, models


# Function designed to read the images from a file given by <path> and return them as a numpy array
def readImages(path):
    images = []
    for eachImage in glob.glob(path):
        images.append(image.imread(eachImage))
    images = np.array(images)
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

# Function designed to calculate the accuracy of a model by calculating correct_predictions/total_pred.
def checkPredictions(predict, labels):
    nr = 0
    for i in range(len(predict)):
        if predict[i] != labels[i]:
            nr += 1
    return (len(predict) - nr) / len(predict)


# Writes the ouput in the file 'submission.txt' as required by the Kaggle Competition.
def writePredFile(pre):
    w = open("submission.txt", "w")
    w.write("id,label\n")
    for i in range(35001, 40001):
        linie = "0" + str(i) + ".png," + str(pre[i - 35001]) + "\n"
        w.write(linie)
    w.close()

# This will be the images and the labels with witch I will train my classifier.
train_images = readImages("./train/*.png")
train_labels = readLabels("./train.txt")

# This is the first set of data where I will check how well does my classifier work after training.
validation_images = readImages("./validation/*.png")
validation_labels = readLabels("./validation.txt")

# This is the testing set
test_images = readImages("./test/*.png")

# I need to reshape the images because my keras layers require the input to have 4 dimensions.
# The input should have the following shape : (nr_of_images, length, height, color_chanels)
train_images = train_images.reshape(30001, 32, 32, 1)
validation_images = validation_images.reshape(5000, 32, 32, 1)
test_images = test_images.reshape(5000, 32, 32, 1)


# A keras Sequential model is model with multiple layers one on top of the other,
# where each layer has exactly one input tensor and one output tensor.
model = models.Sequential([
    # A convolutional layer that receives the input of 32x32x1 with:
    # 'relu' activation, 32 filters and (5x5) size of a filter
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 1)),
    # Takes the result for convolution and moves a 2x2 pooling filter over the matrix and takes only the max value
    # from the 2x2 matrix scanned, resulting in an image with width and height halfed.
    layers.MaxPooling2D((2, 2)),

    # 'relu' activation, 64 filters and (5x5) size of a filter
    layers.Conv2D(64, (5, 5), activation='relu'),
    # Takes the result for convolution and moves a 2x2 pooling filter over the matrix and takes only the max value
    # from the 2x2 matrix scanned, resulting in an image with width and height halfed.
    layers.MaxPooling2D((2, 2)),

    # 'relu' activation, 128 filters and (5x5) size of a filter
    layers.Conv2D(128, (5, 5), activation='relu'),

    # Changes the input of features from multidimensional to one-dimension.
    layers.Flatten(),

    # Compresses the features received from the last layer in 128 units.
    layers.Dense(128, activation='relu'),
    # Switches off a part of the neurons to try to avoid overfitting.
    layers.Dropout(0.2),
    # Compresses the features in 9 because that is the number of classes that I want my model to output. (0..8)
    # Softmax is required when you have more then 2 possible labels.
    layers.Dense(9, activation='softmax')
])

# Compiles the model with adam optimizer (stochastic gradient descent),
# with loss set to SparseCategoricalCrossentropy(9 label classes)
# and metrics = ['accuracy'] which calculates how often predictions equal labels.
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Calls the fit on train_images and train_labels, with the number of samples before each update equal to 128
# and the number of passes though the training set equal to 16.
# Validation data is also given to help as estimate accuracy and the loss of the model on the validation data
# after each epoch is run.
model.fit(train_images, train_labels,
          batch_size=128,
          epochs=16,
          validation_data=(validation_images, validation_labels))

# Saves the predictions of the model on the validation images.
predictions = model.predict_classes(validation_images)
# Writes a classification_report about the predicitions comparing them with the corect ones.
print(classification_report(validation_labels, predictions))
print(classification_report(validation_labels, predictions))

# s the predictions of the model on the test images and write them using the predifined function.
predictions = model.predict_classes(test_images)
writePredFile(predictions)