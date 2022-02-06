import tensorflow_datasets as tfds
import matplotlib.pyplot as plt




def frame_show(ds, Frame_No=0, with_segment=True):
    # ds = ds.take(1)
    print(type(ds))
    print(ds['video']['frames'])
    print(ds['video']['frames'].numpy().shape)

    video = ds['video']['frames']
    print(type(video))
    if with_segment:
        fig, axs = plt.subplots(2)
        segment = ds['video']['segmentations']
        axs[1].imshow(segment[Frame_No, :, :, 0], interpolation='nearest')
        axs[0].imshow(video[Frame_No, :, :, :], interpolation='nearest')
    else:
        plt.show(video[Frame_No, :, :, :], interpolation='nearest')

    plt.show()