import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


def frame_show(ds, video_no=0, frame_no=0, with_segment=True):
    """
    Ploting the image and its segmentation
    :param ds: The dataset
    :param video_no: The number of place the in the row
    :param frame_no: Number of the frame
    :param with_segment: Plot the segmentation alogn video or not

    TODO: DO IT FOR OTHER DATASETS
          USE OTHER LIBRARY TO SHOW THE IMAGE SUCH AS PILLOW Library

    """
    def _plot(num):
        video = ds[num]
        if with_segment:
            fig, axs = plt.subplots(2)
            frames = video['x']
            segments = video['y']
            axs[1].imshow(segments[frame_no, :, :, 0], interpolation='nearest')
            axs[0].imshow(frames[frame_no, :, :, :], interpolation='nearest')
        else:
            plt.show(video[frame_no, :, :, :], interpolation='nearest')

        plt.show()

    ds = list(ds)

    if isinstance(video_no, list):
        for num in video_no:
            _plot(num)

    else:
        _plot(video_no)




def shape_extractor(ds):
    """
    Return shapes of a dataset
    :param ds: a prefetch dataset object
    :return: shape of frames, shape of segmentation

    TODO: Implement it for other datasets
    """
    indim_shape = ds.take(1)
    for example in tfds.as_numpy(indim_shape):
        return example['x'].shape[1:], \
               example['y'].shape[1:]


