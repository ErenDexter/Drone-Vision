import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


image = tf.keras.preprocessing.image
load = tf.keras.models.load_model


model_path = 'CNN.keras'
model = load(model_path)

img_path = 'Dataset\Images_Validation\ME_414\ME 414 (7).jpg'
img = image.load_img(img_path, target_size=(256, 256))

img = image.img_to_array(img)
img /= 255.0
img = img[np.newaxis, ...] 

target_layer_name = 'max_pooling2d_19'

layer_output = model.get_layer(target_layer_name).output
feature_maps_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)


feature_maps = feature_maps_model.predict(img)

feature_maps = np.squeeze(feature_maps)
num_channels = feature_maps.shape[-1]
feature_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min())

num_rows = int(np.ceil(np.sqrt(num_channels)))
num_cols = int(np.ceil(num_channels / num_rows))

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))
axes = axes.ravel()
for i in range(num_channels):
    ax = axes[i]
    ax.imshow(feature_maps[:, :, i], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()