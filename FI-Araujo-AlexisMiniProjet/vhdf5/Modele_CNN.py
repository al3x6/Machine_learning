import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tqdm import tqdm

# Chargement des données
dataset_train = "datasets/train_set.hdf5"
dataset_test = "datasets/test_set.hdf5"

# Chargement du dataset d'entraînement
dataset = h5py.File(dataset_train, "r")
x_train = np.array(dataset["X"][:])
y_train = np.array(dataset["y"][:])
dataset.close()

# Chargement du dataset de test
dataset = h5py.File(dataset_test, "r")
x_test = np.array(dataset["X"][:])
y_test = np.array(dataset["y"][:])
dataset.close()

# Affichage des dimensions des données d'entraînement et de test
print(x_train.shape)
y_train = y_train.reshape(y_train.shape[0], 1)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))

print(x_test.shape)
y_test = y_test.reshape(y_test.shape[0], 1)
print(y_test.shape)
print(np.unique(y_test, return_counts=True))

# Réorganisation et normalisation des données
x_train_reshape = x_train.reshape(x_train.shape[0], -1) / x_train.max()
x_test_reshape = x_test.reshape(x_test.shape[0], -1) / x_train.max()

# Définition du modèle de réseau de neurones multicouches
def build_multilayer_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Entraînement du modèle
def train_multilayer_model(x_train, y_train, x_test, y_test, n_iter=100):
    input_shape = x_train.shape[1:]
    multilayer_model = build_multilayer_model(input_shape)

    train_cout = []
    train_accuracy = []
    test_cout = []
    test_accuracy = []

    for i in tqdm(range(n_iter)):
        history = multilayer_model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        train_cout.append(history.history['loss'][0])
        train_accuracy.append(history.history['accuracy'][0])

        test_loss, test_acc = multilayer_model.evaluate(x_test, y_test, verbose=0)
        test_cout.append(test_loss)
        test_accuracy.append(test_acc)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_cout, label='train_cout')
    plt.plot(test_cout, label='test_cout')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='train_accuracy')
    plt.plot(test_accuracy, label='test_accuracy')
    plt.legend()
    plt.show()

    return multilayer_model

# Entraînement du modèle avec les données chargées
trained_model = train_multilayer_model(x_train_reshape, y_train, x_test_reshape, y_test)

# Fonction pour prédire une seule image avec le modèle entraîné
def predict_single_image(image_path, model, image_size=(64, 64)):
    img = Image.open(image_path).convert('L').resize(image_size)
    img_array = np.array(img).reshape(1, -1) / 255.0
    prediction = model.predict(img_array)

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    if prediction >= 0.5:
        print("Rose", prediction)
    else:
        print("Tulipe", prediction)

# Utilisation de la fonction pour prédire une image avec le modèle entraîné
image_path_to_predict = 'donnees_basique/rose1.jpg'
predict_single_image(image_path_to_predict, trained_model)