
# @misc{TFDS,
#   title = { {TensorFlow Datasets}, A collection of ready-to-use datasets},
#   howpublished = {\url{https://www.tensorflow.org/datasets}},
# }

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt




# @tfds.decode.make_decoder()
# def decode_example(serialized_image, feature):
#   crop_y, crop_x, crop_height, crop_width = 20, 20, 20, 3
#   return tf.image.decode_and_crop_jpeg(
#       serialized_image,
#       [crop_y, crop_x, crop_height, crop_width],
#       channels=feature.feature.shape[-1],
#   )


def load_dataset(name, split='train'):
    """
        davis dataset structure
    FeaturesDict({
     'metadata': FeaturesDict({
         'num_frames': tf.int64,
         'video_name': tf.string,
      }),
     'video': Sequence({
         'frames': Image(shape=(None, None, 3), dtype=tf.uint8),
         'segmentations': Image(shape=(None, None, 1), dtype=tf.uint8),
      }),
    })

    :param split: it should be 'test' or 'train'
    :param name: name of a datasets.

    :return: loding_dataset the dataset in federated mode for Training, or
             simple sequence dataset for Testing Phase
    """
    ds, _ = tfds.load(name=name, split=split, with_info=True, download=False)
    print(ds)
    return ds



def metadata_extractor(ds):
    for example in tfds.as_numpy(ds):
        return (example['metadata']['num_frames'], # number of frames
                example['metadata']['video_name']) # objects in video

def pad(ds):
    """
    Resize with crop or pad the data,  the second dimension
    pass: (80, 480, 854, 3) -> (80, 480, 854, 3)
    pad:  (80, 480, 854, 3) -> (80, 480, 854, 3)
    crop: (76, 480, 938, 3) -> (76, 480, 854, 3)

    TODO:
        shape condition for other dataset
    """
    def padding(ele):
        ele['video']['frames'] = tf.image.resize_with_crop_or_pad(ele['video']['frames'], 480, 854)
        ele['video']['segmentations'] = tf.image.resize_with_crop_or_pad(ele['video']['segmentations'], 480, 854)
        return ele


    return ds.map(lambda x: padding(x))

