import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
from Siamese_Model.Layers.layers import EmbeddingBlock, L1Dist


class Siamese(keras.Model):
    def __init__(self, num_classes=1, **kwargs):
        super(Siamese, self).__init__(**kwargs)

        self.input_embedding_block = EmbeddingBlock()
        self.validation_embedding_block = EmbeddingBlock()
        self.dist = L1Dist()
        self.classifier = Dense(num_classes, activation='sigmoid')

    def call(self, input_tensor, validation_tensor, training=False):
        input_embedding = self.input_embedding_block(input_tensor, training=False)
        validation_embedding = self.validation_embedding_block(validation_tensor, training=False)

        x = self.dist(input_embedding, validation_embedding)
        x = Flatten()(x)
        x = self.classifier(x)

        return x
