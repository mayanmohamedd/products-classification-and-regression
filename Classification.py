import warnings
import os
import cv2
import numpy as np
from keras.src.optimizers.adam import Adam
from sklearn.metrics import accuracy_score


warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.cluster import MiniBatchKMeans

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".iCCP: known incorrect sRGB profile.", module="PIL")


import random
random.seed(5)
np.random.seed(5)


def reading_images(folder_path, type, size=(224, 224)):
    Data = {}
    if os.path.exists(folder_path):
        # Loop through each file in the folder
        if 'Classification' in folder_path:

            for filename in os.listdir(folder_path):
                # print(filename)
                images = []
                FP = os.path.join(folder_path, filename)
                # print(FP)
                # for files in os.listdir(FP):

                if (type == 'Train'):  # & files.find('Train') != -1:
                    file_path = os.path.join(FP, 'Train')
                    # print(file_path)
                elif (type == 'Validation'):  # & files.find('Validation') != -1:
                    file_path = os.path.join(FP, 'Validation')
                for image in os.listdir(file_path):
                    image_path = os.path.join(file_path, image)

                    img = cv2.imread(image_path, 0)
                    img = cv2.resize(img, size)

                    images.append(img)
                # print(file_path, len(images))
                Data[filename] = images
        else:

            for subFolder in os.listdir(folder_path):

                if (type == 'Train' and 'Training' in subFolder):
                    file_path = os.path.join(folder_path, 'Training Data')
                    # print(file_path)
                    for f in os.listdir(file_path):
                        file = os.path.join(file_path, f)
                        # print(f)
                        images = []
                        for image in os.listdir(file):
                            image_path = os.path.join(file, image)
                            # print(i)
                            img = cv2.imread(image_path, 0)
                            img = cv2.resize(img, size)

                            images.append(img)
                        # print(f , len(images))
                        Data[file] = images

                elif (type == 'Validation' and 'Validation' in subFolder):
                    file_path = os.path.join(folder_path, 'Validation Data')
                    # print(file_path)
                    for f in os.listdir(file_path):
                        file = os.path.join(file_path, f)
                        # print(f)
                        images = []
                        for image in os.listdir(file):
                            image_path = os.path.join(file, image)
                            # print(i)
                            img = cv2.imread(image_path, 0)
                            img = cv2.resize(img, size)

                            images.append(img)
                        # print(f , len(images))
                        Data[file] = images
    return Data



class_folderPath = 'C:\\Users\\PC\\PycharmProjects\\pythonProject11\\Data\\Product Classification'
Classification_Train = reading_images(class_folderPath, 'Train')
X_train_alex = []
y_train_alex = []

for class_name, images in Classification_Train.items():
    X_train_alex.extend(images)  # Add images to X_train
    y_train_alex.extend([class_name] * len(images))

X_train_alex_array = np.array(X_train_alex)
y_train_alex_array = np.array(y_train_alex)


X_train_alex_array = np.expand_dims(X_train_alex_array, axis=-1)  # Add channel dimension
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_alex_array)

# Convert integer labels to one-hot encoded vectors
onehot_encoder = OneHotEncoder(sparse=False)
y_train_onehot = onehot_encoder.fit_transform(y_train_encoded.reshape(-1, 1))


Classification_Validation = reading_images(class_folderPath, 'Validation')
X_val_alex = []
y_val_alex = []

for class_name, images in Classification_Validation.items():
    X_val_alex.extend(images)
    y_val_alex.extend([class_name] * len(images))

# Convert the lists to NumPy arrays
X_val_alex_array = np.array(X_val_alex)
y_val_alex_array = np.array(y_val_alex)

# Reshape the image data if necessary
X_val_alex_array = np.expand_dims(X_val_alex_array, axis=-1)  # Add channel dimension if needed

# Encode labels for validation data
y_val_encoded = label_encoder.transform(y_val_alex_array)
y_val_onehot = onehot_encoder.transform(y_val_encoded.reshape(-1, 1))




Recog_folderPath = 'C:\\Users\\PC\\PycharmProjects\\pythonProject11\\Data\\Product Recoginition'
Recoginition_Train = reading_images(Recog_folderPath, 'Train')
Recoginition_Validation = reading_images(Recog_folderPath, 'Validation')




def extract_sift_features(images):
    sift = cv2.SIFT_create()
    all_descriptors = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            all_descriptors.extend(descriptors)
    return np.array(all_descriptors)


def create_vocabulary(descriptors, k=100):
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=3072)
    kmeans.fit(descriptors)
    return kmeans

def image_to_histogram(image, vocabulary):
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(image, None)
    histogram = np.zeros(len(vocabulary.cluster_centers_))
    if descriptors is not None:
        words = vocabulary.predict(descriptors)
        for w in words:
            histogram[w] += 1
    return histogram

# Extract descriptors for training images
all_descriptors = []
for images in Classification_Train.values():
    all_descriptors.extend(extract_sift_features(images))

# Create BoW vocabulary
vocabulary = create_vocabulary(np.array(all_descriptors), k=100)

# Convert each image to a histogram feature vector
for category, images in Classification_Train.items():
    Classification_Train[category] = [image_to_histogram(img, vocabulary) for img in images]

from sklearn import svm


# Prepare data for training
X_train = []
y_train = []

for category, histograms in Classification_Train.items():

    X_train.extend(histograms)
    y_train.extend([category] * len(histograms))

# Convert each image to a histogram feature vector using the vocabulary
X_val_hist = []
for category, images in Classification_Validation.items():
    histograms = [image_to_histogram(img, vocabulary) for img in images]
    X_val_hist.extend(histograms)

# Create true labels for the validation set
y_val_true = []
for category, histograms in Classification_Validation.items():
    y_val_true.extend([category] * len(histograms))
print("y_val_true",y_val_true)

# Initialize SVM classifier
clf = svm.SVC(kernel='linear')  # You can change the kernel as needed

# Train the SVM classifier
clf.fit(X_train, y_train)

# Predict labels for the validation data histograms
predictions = clf.predict(X_val_hist)


# Calculate accuracy on the validation set
accuracy = accuracy_score(y_val_true, predictions)
print("Validation set accuracy of default svm:", accuracy)





from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=250)  # You can adjust parameters as needed

# Train the Random Forest Classifier
rf_classifier.fit(X_train, y_train)

# Predict labels for the validation data histograms using Random Forest
rf_predictions = rf_classifier.predict(X_val_hist)

# Calculate accuracy on the validation set using Random Forest
rf_accuracy = accuracy_score(y_val_true, rf_predictions)
print("Random Forest Validation set accuracy:", rf_accuracy)

num_classes = len(np.unique(y_train))




import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define AlexNet architecture using Keras layers
model = Sequential([
    Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D((3, 3), strides=(2, 2)),
    Conv2D(256, (5, 5), activation='relu'),
    MaxPooling2D((3, 3), strides=(2, 2)),
    Conv2D(384, (3, 3), activation='relu'),
    Conv2D(384, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((3, 3), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Adjust num_classes based on your number of classes
])



optimizer = Adam(learning_rate=0.0001)  # Learning rate decay over each update
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(X_train_alex_array, y_train_onehot, epochs=50, batch_size=32)
test_loss, test_accuracy = model.evaluate(X_val_alex_array, y_val_onehot, verbose=3)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
