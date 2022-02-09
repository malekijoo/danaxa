import tensorflow as tf


class ResBlock(tf.keras.Model):

  def __init__(self, kernel_size, filters):

    super(ResBlock, self).__init__(name='')

    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):

    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor

    return tf.nn.relu(x)


class Encoder(tf.keras.Model):

  def __init__(self, latent_dim):

    super(Encoder, self).__init__()

    self.latent_dim = latent_dim
    self._conv2d = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu')
    self._flatten = tf.keras.layers.Flatten()
    self._dense = tf.keras.layers.Dense(latent_dim, activation='relu')


  def call(self, x):

    x = self._conv2d(x)
    x = self._conv2d(x)
    x = self._flatten(x)
    x = self._dense(x)

    return x


class Decoder(tf.keras.Model):

  def __init__(self, outdim, latent_dim):

    super(Encoder, self).__init__()




  def call(self, x):

    x = self._conv2d(x)
    x = self._conv2d(x)
    x = self._flatten(x)
    x = self._dense(x)

    return x








class VOS_Model(tf.keras.Model):

  def __init__(self):
    print('Space-time Memory Networks: is initializing.')

    super().__init__()

    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)


# keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

