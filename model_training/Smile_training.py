# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import np_utils
from imutils import paths
import imutils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


# initialize the list of data and labels
data = []
labels = []

# looping over the input ismages
for imagePath in sorted(list(paths.list_images("dataset"))):# "dataset" is the folder containing the training named dataset
    # load the image, pre-process it, and store in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    # extravt the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-3]
    label = 'smiling' if label == 'positives' else 'not_smiling'
    labels.append(label)


data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# converting the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['accuracy'])

# training the network
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=classWeight, batch_size=64, epochs=35, verbose=1)

# evaluating the network
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# saving the model to disk
model.save("model/trained_model.h5")

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 35), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 35), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 35), H.history['acc'], label='acc')
plt.plot(np.arange(0, 35), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()