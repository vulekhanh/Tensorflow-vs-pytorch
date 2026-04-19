import argparse
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["accuracy", "time"])
    parser.add_argument("--target_acc", type=float, default=0.55)
    parser.add_argument("--time_limit", type=int, default=600)
    args = parser.parse_args()

    MODE = args.mode
    TARGET_ACC = args.target_acc
    TIME_LIMIT = args.time_limit

    # load data
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    num_classes = 100
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    def preprocess(image, label):
        image = tf.image.resize(image, (224, 224))
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(10000) \
        .map(preprocess) \
        .batch(64) \
        .prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        .map(preprocess) \
        .batch(64) \
        .prefetch(tf.data.AUTOTUNE)

    # simple ResNet-like model
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    start_time = time.time()
    results = []

    for epoch in range(50):
        print(f"\nEpoch {epoch+1}")

        history = model.fit(train_ds, validation_data=test_ds, epochs=1, verbose=1)

        acc = history.history['val_accuracy'][0]
        elapsed = time.time() - start_time

        print(f"Accuracy: {acc:.4f} | Time: {elapsed:.2f}s")

        results.append([epoch+1, acc, elapsed])

        if MODE == "accuracy" and acc >= TARGET_ACC:
            print("Reached target accuracy")
            break

        if MODE == "time" and elapsed >= TIME_LIMIT:
            print("Reached time limit")
            break

    df = pd.DataFrame(results, columns=["Epoch", "Accuracy", "Time"])
    df.to_csv(f"results_tf_{MODE}.csv", index=False)

    print("Saved results!")

if __name__ == "__main__":
    main()
