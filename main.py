from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shutil
from progress.bar import Bar
from helper import *


base_path = os.getcwd()
path = os.getcwd() + r"/data"
os.chdir(path)

painting_filenames = []

with os.scandir(path) as files:
    for file in files:
        if file.name.endswith('.jpg'):
            painting_filenames.append(file.name)

       
model = VGG16()
extracted_features_model  = Model(inputs = model.inputs, outputs = model.layers[-2].output)

painting_features = {}

progress_bar = Bar('Extracting features', max=len(painting_filenames))
for filename in painting_filenames:
    features = extract_features(filename, extracted_features_model)
    painting_features[filename] = features
    progress_bar.next()
 
filenames = np.array(list(painting_features.keys()))
features = np.array(list(painting_features.values()))
features = features.reshape(-1,4096)

pca = PCA(n_components=100, random_state=22)
pca.fit(features)
transformed_features = pca.transform(features)

kmeans = KMeans(n_clusters=20, random_state=22)
kmeans.fit(transformed_features)

groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)  
   
sse = []
list_k = list(range(3, 50))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(transformed_features)
    
    sse.append(km.inertia_)

os.chdir(base_path)

pathResult = os.getcwd() + r"/result"
os.mkdir(pathResult)

for key in groups:
  os.mkdir(pathResult + "/" + str(key))
  for image in groups[key]:
    shutil.copy(path + r"/" + image, pathResult + r"/" + str(key))
