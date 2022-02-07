import tensorflow_datasets as tfds
import matplotlib.pyplot as plt




def frame_show(ds, video_no=0, Frame_No=0, with_segment=True):

    ds = list(ds)
    video = ds[video_no]
    print(type(video))
    if with_segment:
        fig, axs = plt.subplots(2)
        frames = video['video']['frames']
        segments = video['video']['segmentations']
        axs[1].imshow(segments[Frame_No, :, :, 0], interpolation='nearest')
        axs[0].imshow(frames[Frame_No, :, :, :], interpolation='nearest')
    else:
        plt.show(video[Frame_No, :, :, :], interpolation='nearest')

    plt.show()