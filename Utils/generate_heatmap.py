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

layer_outputs = []

layer_names = [layer.name for layer in model.layers]
for layer_name in layer_names:
    layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    layer_output = layer_model.predict(img)
    layer_outputs.append(layer_output)


layer_outputs = [output / np.max(output) for output in layer_outputs]

fig_size = (12, 8)  
for i, layer_name in enumerate(layer_names):
    heatmap = layer_outputs[i]
    
    if len(heatmap.shape) == 4:  
        heatmap = np.mean(heatmap, axis=-1)  
        heatmap = np.squeeze(heatmap, axis=0)  
    elif len(heatmap.shape) == 1:
        heatmap = np.reshape(heatmap, (1, -1))
    elif len(heatmap.shape) == 3:
        heatmap = np.squeeze(heatmap, axis=0)
    elif len(heatmap.shape) == 2:
        heatmap = np.expand_dims(heatmap, axis=-1)
    
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(heatmap, cmap='viridis')
    ax.set_title(layer_name)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax)
    plt.show()