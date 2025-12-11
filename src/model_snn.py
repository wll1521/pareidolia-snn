import tensorflow as tf
from tensorflow.keras import layers, Model
from .preprocess import img_size 

####### Siamese Neural Network Model ######
# 1 Variable: Can change margin for required distance between dissimilar pairs
# using Contrastive loss which pushes apart dissimilar pairs and pulls together similar ones
# Can also alter layers or change to another loss metric for a new model entirely

INPUT_SHAPE=(img_size, img_size, 3)

def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

def contrastive_loss(y_true, y_pred, margin=1.0):
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

def build_snn(input_shape=INPUT_SHAPE):
    base_cnn = tf.keras.Sequential([
        layers.Conv2D(64, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3)
    ])

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    feat_a = base_cnn(input_a)
    feat_b = base_cnn(input_b)

    distance = layers.Lambda(euclidean_distance)([feat_a, feat_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    model.compile(loss=contrastive_loss, optimizer=tf.keras.optimizers.Adam(1e-4))
    return model
