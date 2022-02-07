import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def frame_show(ds, video_no=0, frame_no=0, with_segment=True):

    ds = list(ds)
    video = ds[video_no]
    if with_segment:
        fig, axs = plt.subplots(2)
        frames = video['video']['frames']
        segments = video['video']['segmentations']
        axs[1].imshow(segments[frame_no, :, :, 0], interpolation='nearest')
        axs[0].imshow(frames[frame_no, :, :, :], interpolation='nearest')
    else:
        plt.show(video[frame_no, :, :, :], interpolation='nearest')

    plt.show()