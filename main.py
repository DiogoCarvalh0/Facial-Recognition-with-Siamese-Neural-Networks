import os

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from Siamese_Model.siameseModel import Siamese
import Configs.config as config
from Utils.preprocess import *
from Utils.training_loop import *


def main():
    positive = tf.data.Dataset.list_files(f'{config.POSITIVE_LABEL_PATH}*.jpg').take(config.NR_IMAGES_TO_USE)
    negative = tf.data.Dataset.list_files(f'{config.NEGATIVE_LABEL_PATH}*.jpg').take(config.NR_IMAGES_TO_USE)
    anchor = tf.data.Dataset.list_files(f'{config.ANCHOR_LABEL_PATH}*.jpg').take(config.NR_IMAGES_TO_USE)

    # Read and concatenate data
    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    # Build DataLoader pipeline
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=3 * config.NR_IMAGES_TO_USE)

    # Training partition
    train_data = data.take(round(len(data) * 0.7))
    train_data = train_data.batch(config.BATCH_SIZE)
    train_data = train_data.prefetch(8)

    # Validation partition
    validation_data = data.skip(round(len(data) * 0.7))
    validation_data = data.take(round(len(data) * 0.3))
    validation_data = validation_data.batch(config.BATCH_SIZE)
    validation_data = validation_data.prefetch(8)

    data = (train_data, validation_data)

    # Instantiate Model
    siamese_model = Siamese()
    # Instantiate Loss function
    binary_cross_loss = tf.keras.losses.BinaryCrossentropy()
    # Instantiate Optimizer
    optimizer = tf.keras.optimizers.Adam(3e-4)
    # Metric
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_recall_metric = tf.keras.metrics.Recall()
    val_precision_metric = tf.keras.metrics.Precision()
    val_auc_metric = tf.keras.metrics.AUC()

    metrics = {
        'accuracy': train_acc_metric,
        'val_accuracy': val_acc_metric,
        'val_recall': val_recall_metric,
        'val_precision': val_precision_metric,
        'val_auc': val_auc_metric,
    }
    # tensorboard writer
    tb_writer = tf.summary.create_file_writer(f'./logs/{config.MODEL_NAME}')

    train(data, siamese_model, binary_cross_loss, optimizer, metrics, config.EPOCHS, tb_writer)

    siamese_model.save(f'./saved-model/{config.MODEL_NAME}')


if __name__ == '__main__':
    main()
