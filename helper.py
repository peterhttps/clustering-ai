from keras.preprocessing.image import load_img 
from keras.applications.vgg16 import preprocess_input 
import numpy as np

def extract_features(image_file, model):
    image = load_img(image_file, target_size=(224,224))
    image_array = np.array(image) 
    reshaped_image_array = image_array.reshape(1,224,224,3) 
    preprocessed_image = preprocess_input(reshaped_image_array)
    features = model.predict(preprocessed_image, use_multiprocessing=True)
    return features 
  