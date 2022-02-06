import os
import utils
import dataset
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2


import matplotlib.pyplot as plt


ds = dataset.load_dataset(name="davis", split="train")
print(ds)

first = next(ds)
print(first['video']['frames'].numpy().shape)
utils.frame_show(first)
first = dataset.pad(first['video'])
print(first['video']['frames'].numpy().shape)


first = next(ds)
print(first['video']['frames'].numpy().shape)

first = next(ds)
print(first)
print(first['video']['frames'].numpy().shape)
# for example in
# print(ds.padded_batch(10))
# print(len(ds))
# ds = ds.take(5)
# ds = ds.batch(60)
# print(len(ds))
# print(len(ds[0]))
# print(info.features['video']['frames'].dtype)
# print(info.features['metadata']['video_name'])
# print(dataset.metadata_extractor(ds))
# dataset.frame_show(ds, Frame_No=30)
# dt = iter(ds)
# print(dt['video']['frames'].numpy().shape)
# print(next(dt)['video']['frames'].numpy().shape)
# print(next(dt)['video']['frames'].numpy().shape)
for example in ds:
    print(example['video']['frames'].numpy().shape)
    if example['video']['frames'].numpy().shape[2] != 854:
        print('not 854')









def train():
    pass

def test():
    pass




if __name__ == '__main__':
    pass
    # print('Video Object Segmentation')
