import os
import numpy as np
from PIL import Image

image_dir = 'Random Forest\Dataset'

target_size = (256, 256)  

images = []
labels = []

for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path)

        img = img.resize(target_size)

        label = "_".join([filename.split(' ')[0], filename.split(' ')[1]]) 
        print(label)

        img_array = np.array(img).flatten()

        images.append(img_array)
        labels.append(label)

X = np.array(images)
y = np.array(labels)

np.save('Random Forest\Processed_Dataset\X.npy', X)
np.save('Random Forest\Processed_Dataset\y.npy', y)