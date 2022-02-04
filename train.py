import tensorflow as tf
from tensorflow import keras
from colorama import Fore, Back, Style

print(Fore.GREEN + "All modules loaded successfuly!" + Style.RESET_ALL)

dataset = tf.keras.datasets.mnist

print(Fore.GREEN + "Dataset loaded successfuly!" + Style.RESET_ALL)

(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.

classification_model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

print(Fore.GREEN + "Model created successfuly! (input: (28, 28), hidden layers: 128, output layers: 10)" + Style.RESET_ALL)

classification_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(Fore.GREEN + "Model compiled successfuly!" + Style.RESET_ALL)

classification_model.fit(x_train, y_train, epochs=5)
classification_model.evaluate(x_test,  y_test)

print(Fore.GREEN + "Model trained successfuly!" + Style.RESET_ALL)

classification_model.save("hwd_model.h5")

print(Fore.GREEN + "Model saved successfuly! (hwd_model.h5)" + Style.RESET_ALL)
print(Fore.GREEN + "\n========\nFINISHED!\n" + Style.RESET_ALL)
