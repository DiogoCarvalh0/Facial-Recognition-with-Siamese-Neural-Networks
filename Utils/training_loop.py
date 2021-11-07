import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


@tf.function
def train_step(batch, model, loss, optimizer, metrics):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        predictions = model(X[0], X[1], training=True)
        loss_value = loss(y, predictions)

    # Compute gradient
    grads = tape.gradient(loss_value, model.trainable_weights)
    # Update weights
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # Update metric
    [metric.update_state(y, predictions) for metric in metrics]

    return loss_value


@tf.function
def test_step(data, model, loss, metrics):
    # forward pass, no backprop, inference mode
    print('$'*100)
    print(data.shape)
    X = data[:2]
    y = data[2]
    predictions = model(X[0], X[1], training=False)

    # Compute the loss value
    loss_value = loss(y, predictions)

    # Update val metrics
    [metric.update_state(y, predictions) for metric in metrics]

    return loss_value


def get_train_test_metrics(metrics):
    train_metrics_names = []
    train_metrics_values = []
    val_metrics_names = []
    val_metrics_values = []

    for k, v in metrics.items():
        if 'val' in k:
            val_metrics_names.append(k)
            val_metrics_values.append(v)
        else:
            train_metrics_names.append(k)
            train_metrics_values.append(v)

    return train_metrics_names, train_metrics_values, val_metrics_names, val_metrics_values


def get_metrics_values_for_progbar(metrics_names, metrics_values, values=[]):
    for k, m in zip(metrics_names, metrics_values):
        values.append((k, m.result().numpy()))

    return values


def train(data, model, loss, optimizer, metrics, epochs, tensorboard_writer):
    train_data, val_data = data
    train_metrics_names, train_metrics_values, val_metrics_names, val_metrics_values = get_train_test_metrics(metrics)


    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        progbar = tf.keras.utils.Progbar(len(train_data),
                                         stateful_metrics=['loss', *train_metrics_names, 'val_loss', *val_metrics_names])

        # Reset the metric values for every epoch
        [t_metric.reset_state() for t_metric in train_metrics_values]
        [v_metric.reset_state() for v_metric in val_metrics_values]

        # Train over every batch
        for idx, batch in enumerate(train_data):
            train_loss_value = train_step(batch, model, loss, optimizer, train_metrics_values)

            # Update probbar
            values = [('loss', train_loss_value.numpy())]
            values = get_metrics_values_for_progbar(train_metrics_names, train_metrics_values, values)
            progbar.update(idx+1, values=values)

        # Evaluate model on validation set
        for idx, batch in enumerate(val_data):
            val_loss_value = test_step(batch, model, loss, val_metrics_values)

        # Update probbar to show validation results
        values = [('loss', train_loss_value.numpy())]
        values = get_metrics_values_for_progbar(train_metrics_names, train_metrics_values, values)
        values = [('val_loss', val_loss_value.numpy())]
        values = get_metrics_values_for_progbar(val_metrics_names, val_metrics_values, values)

        progbar.update(len(train_data), values=values)

        # write training loss and accuracy to the tensorboard
        with tensorboard_writer.as_default():
            for record in values:
                tf.summary.scalar(record[0], record[1], step=epoch)
