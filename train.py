#import necessary packages
from module import config
from tensorflow.keras.applications import VGG16 #pip install tensorflow
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer  #pip install sklearn
from sklearn.model_selection import train_test_split
from imutils import paths                         #pip install imutils
import matplotlib.pyplot as plt
import numpy as np                                #pip install numpy
import pickle
import cv2                                       #pip3 install opencv-python
import os

print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []

#loop over csv files in annotation directry
for csvPath in paths.list_files(config.ANNOTS_PATH,validExts=(".csv")):
    #load the contents
    rows = open(csvPath).read().strip().split("\n")
    #loop over the rows:
    for row in rows:
        row = row.split(",")
        (filename, startX, startY, endX, endY, label) = row


        imagePath = os.path.sep.join([config.IMAGES_PATH, label, filename])
        image = cv2.imread(imagePath)
        (h,w) = image.shape[:2]

        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        image = load_img(imagePath, target_size = (224,224))
        image = img_to_array(image)

        data.append(image)
        labels.append(label)
        bboxes.append((startX, startY, endX, endY))
        imagePaths.append(imagePath)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype = "float32")
imagePaths = np.array(imagePaths)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

if len(lb.classes_) ==2:
    labels = to_categorical(labels)

split = train_test_split(data, labels, bboxes, imagePaths, test_size = 0.20, random_state=42)

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

print("[INFO] saving test image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()

vgg = VGG16(weights = "imagenet", include_top = False, input_tensor = Input(shape=(224, 224, 3)))

vgg.trainable = False

flatten = vgg.output
flatten = Flatten()(flatten)

bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)

model = Model(inputs=vgg.input,outputs=(bboxHead,softmaxHead))

losses = {"class_label": "categorical_crossentropy", "bounding_box":"mean_squared_error"}

lossWeights = {"class_label":1.0, "bounding_box":1.0}

opt = Adam(lr=config.INIT_LR)
model.compile(loss = losses, optimizer= opt, metrics = ["accuracy"], loss_weights=lossWeights)
print(model.summary())

trainTargets = {"class_label": trainLabels, "bounding_box": trainBBoxes}

testTargets = {"class_label": testLabels, "bounding_box": testBBoxes}

print("[INFO] trainig model...")
H = model.fit(trainImages, trainTargets, validation_data = (testImages, testTargets), batch_size = config.BATCH_SIZE, epochs= config.NUM_EPOCHS, verbose=1)

print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format = "h5")

print("[INFO] saving label binarizer...")
f = open(config.LB_PATH,  "wb")
f.write(pickle.dumps(lb))
f.close()

