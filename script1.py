import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from keras.applications.inception_v3 import preprocess_input
from test1 import get_encoder, SiameseModel, get_siamese_network, extract_encoder


def read_images_from_folder(root_folder):
    image_list = {}

    products = [product for product in os.listdir(root_folder) if
                os.path.isdir(os.path.join(root_folder, product))]
    products = sorted(products, key=lambda x: int(x))

    for product in products:
        product_directory = os.path.join(root_folder, product)
        image_files = [file for file in os.listdir(product_directory) if file.endswith(('.jpg', '.png'))]

        if image_files:
            for image in image_files:
                image_path = os.path.join(product_directory, image_files[0])
                image = cv2.imread(image_path)
                image_list[product] = image
        else:
            print(f"Warning: No images found for {product}.")

    return image_list


def choose_reference_images(root_directory="Data/Product Recoginition/Training Data"):
    reference_images = {}

    products = [product for product in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, product))]
    products = sorted(products, key=lambda x: int(x))

    for product in products:
        product_directory = os.path.join(root_directory, product)
        image_files = [file for file in os.listdir(product_directory) if file.endswith(('.jpg', '.png'))]

        if image_files:
            image_path = os.path.join(product_directory, image_files[0])
            image = cv2.imread(image_path)
            reference_images[product] = image
        else:
            print(f"Warning: No images found for {product}.")

    return reference_images


def recognize_images(test_images, ref_images):
    siamese_model = SiameseModel(get_siamese_network())
    siamese_model.load_weights("siamese_model-final")

    encoder = extract_encoder(siamese_model)
    encoder.load_weights('encoder')

    display_list = []
    correct_classification = 0
    false_classification = 0
    n = len(test_images)

    for key1, value1 in test_images.items():
        distance_list = []
        labels_list = []
        resized_test_image = cv2.resize(value1, (128, 128), interpolation=cv2.INTER_CUBIC)

        for key2, value2 in ref_images.items():
            resized_ref_image = cv2.resize(value2, (128, 128), interpolation=cv2.INTER_CUBIC)

            tensor1 = encoder.predict(preprocess_input(np.expand_dims(resized_test_image, axis=0)))
            tensor2 = encoder.predict(preprocess_input(np.expand_dims(resized_ref_image, axis=0)))

            dist = np.sum(np.square(tensor1 - tensor2), axis=-1)
            distance_list.append(dist)
            labels_list.append(key2)
            # print(f"distance between {key1} and {key2} is {dist}")

        min_index = distance_list.index(min(distance_list))
        display_list.append(("image", key1, "is product", labels_list[min_index]))
        if labels_list[min_index] == key1:
            correct_classification+=1
        else:
            false_classification+=1

    for i in display_list:
        print(i)
    accuracy = correct_classification/n
    print("The Accuracy is:", accuracy)

    # num_plots = 7
    # f, axes = plt.subplots(num_plots, 2, figsize=(15, 20))
    # i = 0
    # for (key1, value1), (key2, value2) in zip(test_images.items(), ref_images.items()):
    #     axes[i, 0].imshow(value1)
    #     axes[i, 1].imshow(value2)
    #     if i == 6:
    #         break
    #     i += 1
    # plt.show()


train_path = 'Data/Product Recoginition/Training Data'
test_images = read_images_from_folder('test_data1')
ref_images = choose_reference_images(train_path)

recognize_images(test_images, ref_images)