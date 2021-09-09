from keras.applications.vgg19 import VGG19, preprocess_input
import json
import keras
import tensorflow as tf
import vis ## keras-vis
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from vis.utils import utils
from vis.visualization import visualize_saliency
import datetime

def get_saliency(file_name):
    model = VGG19(weights='imagenet')
    model.summary()
    CLASS_INDEX = json.load(open("imagenet_class_index.json"))
    classlabel = []
    for i_dict in range(len(CLASS_INDEX)):
        classlabel.append(CLASS_INDEX[str(i_dict)][1])
    print("N of class={}".format(len(classlabel)))
    _img = load_img(file_name,target_size=(224,224))
    plt.imshow(_img)
    plt.show()

    img = img_to_array(_img)
    img = preprocess_input(img)
    y_pred = model.predict(img[np.newaxis,...])
    class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]
    topNclass = 5
    for i, idx in enumerate(class_idxs_sorted[:topNclass]):
        print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
            i + 1,classlabel[idx],idx,y_pred[0,idx]))
    # Utility to search for layer index by name. 
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, 'predictions')

    model.layers[layer_idx].activation = keras.activations.linear
    model = utils.apply_modifications(model)

    class_idx = class_idxs_sorted[0]
    grad_top1 = visualize_saliency(model,
                               layer_idx,
                               filter_indices = class_idx,
                               seed_input     = img[np.newaxis,...])
    currentDT = datetime.datetime.now()
    out_name = "saliency_map"+str(currentDT) + ".jpg"
    save_name = "./static/images/"+out_name
    plt.imsave(fname=save_name,arr=grad_top1,cmap="plt.cm.hot")
    return out_name
