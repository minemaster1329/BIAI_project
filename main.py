import tensorflow as tf
from tensorflow.python.data.ops.readers import TFRecordDatasetV2

image_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'label_index': tf.io.FixedLenFeature([], tf.int64)
}


class LabelItem:
    index: int
    label: str

    def __init__(self, index, label):
        self.index = index
        self.label = label

    def __str__(self):
        return f'[{self.index}: {self.label}]\n'

    def __repr__(self):
        return str(self)


def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


raw_image_dataset: TFRecordDatasetV2 = tf.data.TFRecordDataset('gs://biai-preprocessing-data/images.tfrecords')
parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

output = []
output_indexes = []
for image_features in parsed_image_dataset:
    label = image_features['label'].numpy().decode("utf-8")
    label_index = image_features['label_index'].numpy()
    if label not in output:
        output.append(label)
        output_indexes.append(label_index)
print(output)
print(output_indexes)

output_label_index = []

for a in range(0, len(output)):
    output_label_index.append(LabelItem(output_indexes[a], output[a]))

output_label_index.sort(key=lambda labelItem: labelItem.index, reverse=False)

print(output_label_index)
# directory = '/media/dane2/BIAI_datasets/main/CropDisease/Crop___DIsease'
#
# BATCH_SIZE = 32
# IMG_DIM = 160
# IMG_SIZE = (IMG_DIM, IMG_DIM)
#
# # creating dataset for training (80% data)
# dataset_training = tensorflow.keras.utils.image_dataset_from_directory(directory,
#                                                                        shuffle=True,
#                                                                        validation_split=0.2,
#                                                                        subset="training",
#                                                                        batch_size=BATCH_SIZE,
#                                                                        seed=42,
#                                                                        image_size=IMG_SIZE
#                                                                        )
#
# # creating dataset for validation (20% data)
# dataset_validation = tensorflow.keras.utils.image_dataset_from_directory(directory,
#                                                                          shuffle=False,
#                                                                          validation_split=0.2,
#                                                                          subset="validation",
#                                                                          batch_size=BATCH_SIZE,
#                                                                          seed=42,
#                                                                          image_size=IMG_SIZE
#                                                                          )
# class_names = dataset_training.class_names
# num_classes = len(class_names)
#
# # prefetching data to improve performance
# AUTOTUNE = tensorflow.data.AUTOTUNE
# dataset_training = dataset_training.cache().prefetch(buffer_size=AUTOTUNE)
# dataset_validation = dataset_validation.cache().prefetch(buffer_size=AUTOTUNE)
#
# # normalizing image color layer (rescaling colors from [0-255] range to [0-1])
# normalization_layer = layers.Rescaling(1.0/255, input_shape=(IMG_DIM, IMG_DIM, 3))
#
# data_augmentation = keras.Sequential(
#   [
#     layers.RandomFlip("horizontal",
#                       input_shape=(IMG_DIM, IMG_DIM, 3)),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#   ]
# )
#
# model = Sequential([
#     data_augmentation,
#     normalization_layer,
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(128, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.3),
#     layers.Flatten(),
#     layers.Dense(256, activation="sigmoid"),
#     layers.Dense(num_classes),
# ])
#
# model.compile(optimizer='adam',
#               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# model.summary()
#
# epochs = 10
# history = model.fit(dataset_training, validation_data=dataset_validation, epochs=epochs)
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(epochs)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
#
# img = keras.utils.load_img("test_photos/rdza_kukurydzy.jpg", target_size=(128, 128))
# img2 = keras.utils.load_img("test_photos/rdza_pszenicy.jpg", target_size=(128, 128))
#
# img_array = keras.utils.img_to_array(img)
# img_array2 = keras.utils.img_to_array(img2)
#
# img_array = tensorflow.expand_dims(img_array, 0)
# img_array2 = tensorflow.expand_dims(img_array2, 0)
#
# prediction = model.predict(img_array)
# prediction2 = model.predict(img_array2)
#
# score = tensorflow.nn.softmax(prediction[0])
# score2 = tensorflow.nn.softmax(prediction2[0])
#
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
#
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score2)], 100 * np.max(score2))
# )
