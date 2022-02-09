import os
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
print(tf.config.list_physical_devices("GPU"))

ds = dataset.load_dataset(name="davis", split="train")
print(ds)
ds = dataset.pad(ds)

model = ml.VOS_Model()




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








def train():
    pass

def test():
    pass




if __name__ == '__main__':
    pass
    # print('Video Object Segmentation')
