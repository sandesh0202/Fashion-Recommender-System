import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from keras.layers import GlobalMaxPooling2D
from keras.applications import ResNet50
import streamlit
from keras.applications.imagenet_utils import preprocess_input


#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.layers import GlobalMaxPooling2D
#from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top = False, input_shape = (224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

streamlit.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except Exception as e:
        print(f"Error saving file: {e}")
        return 0
    
def feature_extraction(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    x, indices = neighbors.kneighbors([features])

    return indices

uploaded_file = streamlit.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        print("File saved successfully!")
        display_image = Image.open(os.path.join("uploads", uploaded_file.name))
        
        # Convert the image to BGR mode
        display_image = display_image.convert("RGB")
        display_image = np.array(display_image)
        display_image = display_image[:, :, ::-1].copy()  # Convert RGB to BGR
        
        streamlit.image(display_image)
        
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        
        indices = recommend(features, feature_list)
        
        cols = streamlit.columns(5)
        col1, col2, col3, col4, col5 = cols

        with col1:
            streamlit.image(Image.open(filenames[indices[0][0]]))
        with col2:
            streamlit.image(Image.open(filenames[indices[0][1]]))
        with col3:
            streamlit.image(Image.open(filenames[indices[0][2]]))
        with col4:
            streamlit.image(Image.open(filenames[indices[0][3]]))
        with col5:
            streamlit.image(Image.open(filenames[indices[0][4]]))
    else:
        streamlit.header("Some error occurred in file upload")