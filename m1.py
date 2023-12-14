import tensorflow as tf
from tensorflow.keras import datasets, layers, initializers, regularizers, constraints
import numpy as np
import matplotlib.pyplot as plt
# Custom Activation Functions
def custom_activation_1(x):
    return tf.math.expm1(tf.math.sin(x))

def custom_activation_2(x):
    return tf.math.expm1(tf.math.cos(x))

def custom_activation_3(x):
    return tf.math.log1p(tf.math.tan(x))

def custom_activation_4(x):
    return tf.math.sqrt(tf.math.softplus(x))

def custom_activation_5(x):
    return tf.math.log(tf.math.cosh(x))

# Custom Initializers
class CustomInitializer1(initializers.Initializer):
    def __call__(self, shape, dtype=None):
        return tf.random.normal(shape, mean=1.0, stddev=0.1, dtype=dtype)

class CustomInitializer2(initializers.Initializer):
    def __call__(self, shape, dtype=None):
        return tf.random.uniform(shape, minval=-0.5, maxval=0.5, dtype=dtype)

class CustomInitializer3(initializers.Initializer):
    def __call__(self, shape, dtype=None):
        return tf.constant(np.random.laplace(loc=0.0, scale=1.0, size=shape), dtype=dtype)

class CustomInitializer4(initializers.Initializer):
    def __call__(self, shape, dtype=None):
        return tf.random.truncated_normal(shape, mean=0.0, stddev=0.1, dtype=dtype)

class CustomInitializer5(initializers.Initializer):
    def __call__(self, shape, dtype=None):
        return tf.constant_initializer(np.random.beta(a=0.5, b=1.0, size=shape))(shape, dtype=dtype)

# Custom Regularizers
def custom_regularizer_1(weights):
    return 0.01 * tf.reduce_sum(tf.sinh(weights))

def custom_regularizer_2(weights):
    return 0.01 * tf.reduce_sum(tf.cosh(weights))

def custom_regularizer_3(weights):
    return 0.01 * tf.reduce_mean(tf.tanh(weights))

def custom_regularizer_4(weights):
    return 0.01 * tf.reduce_sum(tf.atan(weights))

def custom_regularizer_5(weights):
    return 0.01 * tf.reduce_sum(tf.sigmoid(weights))

# Custom Constraints
class CustomConstraint1(constraints.Constraint):
    def __call__(self, w):
        return w * tf.math.sigmoid(w)

class CustomConstraint2(constraints.Constraint):
    def __call__(self, w):
        return tf.clip_by_value(w, -2, 2)

class CustomConstraint3(constraints.Constraint):
    def __call__(self, w):
        return tf.math.sin(w)

class CustomConstraint4(constraints.Constraint):
    def __call__(self, w):
        return tf.math.cos(w)

class CustomConstraint5(constraints.Constraint):
    def __call__(self, w):
        return w - tf.reduce_mean(w)
def custom_loss_1(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(tf.square(y_true - y_pred))


def custom_loss_2(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def custom_loss_3(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.sqrt(y_true + 1e-7) - tf.sqrt(y_pred + 1e-7)))

def custom_loss_4(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred))

def custom_loss_5(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.log1p(y_true) - tf.log1p(y_pred)))

def custom_loss_6(y_true, y_pred):
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred))

def custom_loss_7(y_true, y_pred):
    return tf.reduce_mean(tf.math.squared_difference(tf.nn.softmax(y_true), tf.nn.softmax(y_pred)))

def custom_loss_8(y_true, y_pred):
    return tf.reduce_mean(tf.pow(y_true - y_pred, 4))

def custom_loss_9(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.huber(y_true, y_pred))

def custom_loss_10(y_true, y_pred):
    return tf.reduce_mean(-y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred))
# Load and prepare the MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Build the model
model = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(56, activation=custom_activation_5,
                 kernel_initializer=CustomInitializer5(),
                 kernel_regularizer=custom_regularizer_4,
                 kernel_constraint=CustomConstraint3()),
    layers.Dense(10, activation=custom_activation_2,
                 kernel_initializer=CustomInitializer3(),
                 kernel_regularizer=custom_regularizer_2,
                 kernel_constraint=CustomConstraint4()),

])

# Compile the model with a custom loss function
# Ensure the loss function is compatible with softmax output and one-hot encoded labels
model.compile(optimizer='adam', loss=custom_loss_1, metrics=['accuracy'])

# Train the model and save the training history
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Plotting the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
