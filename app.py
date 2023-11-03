import tensorflow as tf
from keras.utils import load_img, img_to_array
from keras.applications import ResNet50
from keras.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
import os
from keras.preprocessing import image
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape = (224,224,3))
model.trainable = False

model = tf.keras.Sequential([
    model, 
    GlobalMaxPooling2D()
])


def extract_features(img_path, model):
    filepath = os.path.join(img_path)
    img = load_img(filepath, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    pred = model.predict(img).flatten().reshape(-1)
    return pred / np.linalg.norm(pred)

paths_list = ['data/myntradataset/images/{}.jpg'.format(id) for file in os.listdir('data/myntradataset/images')]

feature_list = []

for file in tqdm(paths_list):
    feature_list.append(extract_features(file, model))
 
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(paths_list, open('filenames.pkl', 'wb'))