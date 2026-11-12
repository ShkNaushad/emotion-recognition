"""
Facial Emotion Recognition - Model Training
Dataset : FER2013 (7 emotions, 48x48 grayscale faces)
Note    : Ye script Google Colab (T4 GPU) pe chalayi gayi thi.
          Local PC pe chalane ke liye BASE aur SAVE ke paths badal lo.
"""
import os, json, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE = '/content/fer2013'                          # unzip kiya hua dataset (train/ aur test/)
SAVE = '/content/drive/MyDrive/emotion-project'    # model yahan save hota hai
IMG = (48, 48)
BATCH = 64

common = dict(labels='inferred', label_mode='categorical', color_mode='grayscale',
              image_size=IMG, batch_size=BATCH)

train_ds = keras.utils.image_dataset_from_directory(
    f'{BASE}/train', validation_split=0.1, subset='training', seed=42, **common)
val_ds = keras.utils.image_dataset_from_directory(
    f'{BASE}/train', validation_split=0.1, subset='validation', seed=42, **common)
test_ds = keras.utils.image_dataset_from_directory(
    f'{BASE}/test', shuffle=False, **common)

class_names = train_ds.class_names
print("Classes:", class_names)

# Class weights (softened) taaki disgust jaisi kam images wali class bhi seekhi jaye
counts = np.array([len(os.listdir(f'{BASE}/train/{c}')) for c in class_names])
class_weight = {i: float((counts.sum() / (len(counts) * c)) ** 0.5) for i, c in enumerate(counts)}
print("Class weights:", class_weight)

# Data augmentation (sirf training data pe)
augment = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
])
train_aug = train_ds.map(lambda x, y: (augment(x, training=True), y),
                         num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


def block(x, f):
    for _ in range(2):
        x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(0.25)(x)


inputs = keras.Input((48, 48, 1))
x = layers.Rescaling(1. / 255)(inputs)
for f in (32, 64, 128, 256):
    x = block(x, f)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(f'{SAVE}/emotion_model.keras',
                                    monitor='val_accuracy', save_best_only=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
]

history = model.fit(train_aug, validation_data=val_ds, epochs=60,
                    class_weight=class_weight, callbacks=callbacks)

test_loss, test_acc = model.evaluate(test_ds)
print(f"\nFINAL TEST ACCURACY: {test_acc * 100:.2f}%")   # humara result: 64.78%

with open(f'{SAVE}/class_names.json', 'w') as f:
    json.dump(class_names, f)
with open(f'{SAVE}/history.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
print("Saved: emotion_model.keras, class_names.json, history.json")