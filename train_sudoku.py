import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 3), padding='valid'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_dir = 'D:\\Education\\NN weights\\sudoku_numbers\\train\\'
validation_dir = 'D:\\Education\\NN weights\\sudoku_numbers\\validate\\'

train_datagen = ImageDataGenerator(rescale=1 / 255)
test_datagen = ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(28, 28), class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(28, 28),
                                                        class_mode='categorical')

history = model.fit_generator(train_generator, validation_data=validation_generator, epochs=10, verbose=1)

model.save_weights('sudoku_weights.h5')
model.save('sudoku_model.h5')

