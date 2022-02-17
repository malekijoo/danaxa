import os
import numpy as np
import utils
import dataset
import models as ml
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2


import matplotlib.pyplot as plt


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("Is there a GPU available: "),
print('No GPU is available' if len(tf.config.list_physical_devices("GPU")) == 0 else 'Yes, there is a GPU')

ds = dataset.load_dataset(name="davis", split="train")
print(ds)

# t = ds.take(4)
# for example in tfds.as_numpy(t):
#     print(example['video']['segmentations'].shape, example['video']['frames'].shape)
#
ds = dataset.pad(ds)
print(ds)

t = ds.take(4)
for example in tfds.as_numpy(t):
    print(example['x'].shape, example['y'].shape)
#
# x_indim, _ = utils.shape_extractor(ds)
#
# print(x_indim)
# element = next(iter(ds))
# print(element)

# for example in tfds.as_numpy(ds):
#     print(np.unique(example['video']['segmentations']))
#     print(example['metadata'])
#     print(example['video'])

# utils.frame_show(ds, video_no=[1, 2, 3, 4])


# print(element[])
# print('segmentation label in ', np.unique(element['video']['segmentations'])


# model = ml.VOS_Model(input_x=element, indim=x_indim)
# print(model.summary())



def train():
    pass

def test():
    pass




if __name__ == '__main__':
    pass
    # print('Video Object Segmentation')




# utils.frame_show(ds)

# for example in ds:
#     print(example['video']['frames'].numpy().shape)
#     if example['video']['frames'].numpy().shape[2] != 854:
#         print(example['video']['frames'].numpy().shape)
#         print('not 854 \n')
#         # print(type(example['video']['frames']))
#         # example['video']['frames'] = tf.image.resize_with_crop_or_pad(example['video']['frames'], 480, 854)
#         # print(type(example['video']['frames']))
#         # print('padded', example['video']['frames'].numpy().shape)
#         # print('\n')


