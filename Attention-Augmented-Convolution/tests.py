"""Tests Augmented Attention Layer.

Mainly tests for shape correctness.
"""

import unittest
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np

from AAConv import AAConv

randomItem = np.random.random_sample
getShape = lambda input_: tuple(input_.shape)

class MNISTModel(Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    # self.conv1 = Conv2D(
    #     32, 3, 
    #     activation='relu', 
    #     data_format='channels_first')
    self.conv1 = AAConv(
        channels_out=32,
        kernel_size=3,
        depth_k=16, 
        depth_v=16, 
        num_heads=4)
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# @unittest.skip('Correct')
class TestAAConv(unittest.TestCase):

    @unittest.skip('Correct')
    def test_AAConv(self):

        H = 36
        W = 54
        C_IN = 6
        C_OUT = 53
        B = 3
        kernel_size = 5
        depth_k = 16
        depth_v = 24
        num_heads = 8

        AAlayer = AAConv(
            channels_out=C_OUT,
            kernel_size=kernel_size,
            depth_k=depth_k,
            depth_v=depth_v,
            num_heads=num_heads)

        input_shape = (B, C_IN, H, W)
        AAlayer.build(input_shape)

        # test _split_heads_2d
        x = randomItem((B, depth_k, H, W))
        result = AAlayer._split_heads_2d(x)
        self.assertEqual(
            getShape(result),
            (B, num_heads, H, W, depth_k // num_heads))

        # test _combine_heads_2d
        x = randomItem((B, num_heads, H, W, depth_k // num_heads))
        result = AAlayer._combine_heads_2d(x)
        self.assertEqual(
            getShape(result),
            (B, H, W, depth_k))

        # test _self_attention_2d
        x = randomItem((B, C_IN, H, W))
        result = AAlayer._self_attention_2d(x)
        self.assertEqual(
            getShape(result),
            (B, depth_v, H, W))

        # test call
        x = randomItem(input_shape)
        result = AAlayer(x)
        self.assertEqual(
            getShape(result),
            (B, C_OUT, H, W))

         
    # @unittest.skip('Correct.')
    def test_MNIST_acc(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train[:, tf.newaxis, :, :]
        x_test = x_test[:, tf.newaxis, :, :]

        # import pdb; pdb.set_trace()

        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(10000).batch(32)

        test_ds = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(32)

        model = MNISTModel()

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)

        @tf.function
        def test_step(images, labels):
            predictions = model(images)
            t_loss = loss_object(labels, predictions)

            test_loss(t_loss)
            test_accuracy(labels, predictions)

        EPOCHS = 5

        for epoch in range(EPOCHS):
            for images, labels in train_ds:
                train_step(images, labels)

            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, \
                Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch+1,
                train_loss.result(),
                train_accuracy.result()*100,
                test_loss.result(),
                test_accuracy.result()*100))

            # Reset the metrics for the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

    
    @unittest.skip('Does not yet work.')
    def test_MNIST_acc2(self):
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

        model = tf.keras.models.Sequential([
            # Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            AAConv(
                channels_out=32,
                kernel_size=3,
                depth_k=16, 
                depth_v=16, 
                num_heads=4,
                input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
            ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)

        model.evaluate(x_test,  y_test, verbose=2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Testing Augmented Attention Convolution.')

    parser.add_argument(
        '-E', '--eager', default=False,
        help='Whether to test AAConv.py with eager execution or not.')

    args = parser.parse_args()

    if args.eager:
        tf.config.experimental_run_functions_eagerly(True)

    unittest.main()