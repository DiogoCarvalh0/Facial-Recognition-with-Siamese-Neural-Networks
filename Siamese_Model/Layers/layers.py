import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'	

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size, add_max_pooling_layer=True, pool_size=2, **kwargs):
        super(CNNBlock, self).__init__(**kwargs)

        self.conv = Conv2D(out_channels, kernel_size)
        self.max_pool = MaxPool2D(pool_size)
        self.add_max_pooling_layer = add_max_pooling_layer

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = tf.nn.relu(x)

        if self.add_max_pooling_layer:
            x = self.max_pool(x)

        return x


class EmbeddingBlock(layers.Layer):
    def __init__(self, **kwargs):
        super(EmbeddingBlock, self).__init__(**kwargs)

        self.cnn1 = CNNBlock(out_channels=64, kernel_size=10)
        self.cnn2 = CNNBlock(out_channels=128, kernel_size=7)
        self.cnn3 = CNNBlock(out_channels=128, kernel_size=4)
        self.cnn4 = CNNBlock(out_channels=256, kernel_size=4, add_max_pooling_layer=False)
        self.dense = Dense(4096, activation='sigmoid')

    def call(self, input_tensor, training=False):
        x = self.cnn1(input_tensor, training=training)
        x = self.cnn2(x, training=training)
        x = self.cnn3(x, training=training)
        x = self.cnn4(x, training=training)
        x = Flatten()(x)
        x = self.dense(x)

        return x


class L1Dist(layers.Layer):
    def __init__(self, **kwargs):
        super(L1Dist, self).__init__(**kwargs)

    def call(self, input_tensor, validation_tensor):
        return tf.math.abs(input_tensor - validation_tensor)
