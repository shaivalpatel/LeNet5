import tensorflow as tf
mnist = tf.keras.datasets.mnist


#defining the LeNet5 Model
model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(filters=6, kernel_size=(3,3), activation = 'relu', input_shape=(28,28,1)),
  tf.keras.layers.AveragePooling2D(),
  tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation = 'relu' ),
  tf.keras.layers.AveragePooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units=120, activation='relu'),
  tf.keras.layers.Dense(units=84, activation ='relu'),
  tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#reshaping the dataset for conv layers
x_train=tf.reshape(x_train, [60000,28,28,1])
x_test = tf.reshape(x_test, [10000,28,28,1])


model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)