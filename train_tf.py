import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
from utils import Timer, save_results

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode
num_classes = 100
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Efficient dataset pipeline
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    return image, label

batch_size = 64

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .shuffle(10000) \
    .map(preprocess) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
    .map(preprocess) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

# ResNet-like model
def residual_block(x, filters, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

inputs = layers.Input(shape=(224,224,3))
x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPool2D(3, strides=2, padding='same')(x)

for filters in [64, 128, 256, 512]:
    x = residual_block(x, filters, stride=2)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
timer = Timer()
timer.start()

history = model.fit(train_ds, validation_data=test_ds, epochs=10)

total_time = timer.stop()
final_acc = history.history['val_accuracy'][-1]

print("TensorFlow Time:", total_time)
print("TensorFlow Accuracy:", final_acc)

save_results("TensorFlow", total_time, final_acc)
