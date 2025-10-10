import tensorflow as tf
from tensorflow.keras import layers, Model

def build_snn(input_shape=(128,128,3)):
    base_cnn = tf.keras.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    feat_a = base_cnn(input_a)
    feat_b = base_cnn(input_b)

    distance = layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)))([feat_a, feat_b])
    output = layers.Dense(1, activation='sigmoid')(distance)

    return Model(inputs=[input_a, input_b], outputs=output)
