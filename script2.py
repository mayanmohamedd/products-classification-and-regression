import os
import cv2
import numpy as np
from test1 import get_encoder, SiameseModel, get_siamese_network, extract_encoder
from keras.applications.inception_v3 import preprocess_input


def load_test_images(unseen_folder, anchor_image_path):
    unseen_image_paths = [os.path.join(unseen_folder, file) for file in os.listdir(unseen_folder)
                          if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    unseen_images_dict = {}
    for image_path in unseen_image_paths:
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        unseen_images_dict[image_name] = image

    anchor_image = cv2.imread(anchor_image_path)

    return unseen_images_dict, anchor_image


def test_recognition_of_unseen_products(unseen_folder_path, anchor_image_path):
    siamese_model = SiameseModel(get_siamese_network())
    siamese_model.load_weights("siamese_model-final")

    encoder = extract_encoder(siamese_model)
    encoder.load_weights('encoder')

    unseen_images, anchor_image = load_test_images(unseen_folder_path, anchor_image_path)
    unseen_images.pop('anchor.png')

    resized_anchor_image = cv2.resize(anchor_image, (128, 128), interpolation=cv2.INTER_CUBIC)

    distance_list = []
    labels_list = []
    for key, value in unseen_images.items():
        resized_test_image = cv2.resize(value, (128, 128), interpolation=cv2.INTER_CUBIC)

        tensor1 = encoder.predict(preprocess_input(np.expand_dims(resized_anchor_image, axis=0)))
        tensor2 = encoder.predict(preprocess_input(np.expand_dims(resized_test_image, axis=0)))

        dist = np.sum(np.square(tensor1 - tensor2), axis=-1)
        distance_list.append(dist)
        labels_list.append(key)
        # print(f"distance between {key1} and {key2} is {dist}")

    min_index = distance_list.index(min(distance_list))
    print(f"Anchor image is similar to product {labels_list[min_index]}")


unseen_folder_path = "test_data2"
anchor_image_path = "test_data2/anchor.png"

test_recognition_of_unseen_products(unseen_folder_path, anchor_image_path)
