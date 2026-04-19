import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
from tensorflow.keras import mixed_precision
from utils import Timer, save_results



# =========================
# 2. LOAD DATASET
# =========================
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

num_classes = 100

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# =========================
# 3. DATA AUGMENTATION (CRITICAL FOR ACCURACY)
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
])

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = data_augmentation(image)
    return image, label

# =========================
# 4. OPTIMIZED DATA PIPELINE
# =========================
BATCH_SIZE = 128  # higher = better GPU utilization

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(10000)
train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.cache()
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.cache()
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# =========================
# 5. IMPROVED RESIDUAL BLOCK (ResNet-v1 style)
# =========================
def residual_block(x, filters, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# =========================
# 6. MODEL (balanced depth for CIFAR-100)
# =========================
inputs = layers.Input(shape=(32, 32, 3))

x = layers.Conv2D(64, 3, padding='same', use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# IMPORTANT: balanced downsampling (NOT too aggressive)
for filters in [64, 128, 256]:
    x = residual_block(x, filters, stride=2)

x = residual_block(x, 256, stride=1)

x = layers.GlobalAveragePooling2D()(x)

# force float32 for stability in mixed precision
outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

model = models.Model(inputs, outputs)

# =========================
# 7. OPTIMIZER (important for convergence)
# =========================
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-3,
    weight_decay=1e-4
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 8. TRAINING
# =========================
timer = Timer()
timer.start()

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50   # IMPORTANT (10 epochs is too small)
)

total_time = timer.stop()
final_acc = history.history['val_accuracy'][-1]

print("\nTime:", total_time)
print("Final Accuracy:", final_acc)

save_results("CIFAR100_Strong_Baseline", total_time, final_acc)
