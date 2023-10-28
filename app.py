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
    image_dir = 'image'
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        img = load_img(filepath, target_size=(224,224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        pred = model.predict(img).flatten()
        normalised_result = pred/norm(pred)
    return normalised_result

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))
    
feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))
 
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))