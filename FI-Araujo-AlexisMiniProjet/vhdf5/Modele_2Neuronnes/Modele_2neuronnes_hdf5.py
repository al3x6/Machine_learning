from PIL import Image
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
#from utilities import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import glob
# import cv2

## Importation du dataSet
dataset_train = "../datasets/train_set.hdf5"
dataset_test = "../datasets/test_set.hdf5"

#on load le dataset d'entrainement
dataset = h5py.File(dataset_train, "r")

x_train = np.array(dataset["X"][:])
y_train = np.array(dataset["y"][:])

#on load le dataset de test
dataset = h5py.File(dataset_test, "r")

x_test = np.array(dataset["X"][:])
y_test = np.array(dataset["y"][:])

# Afficher les dimensions des données d'entrainement et des étiquettes
print(x_train.shape)
y_train = y_train.reshape(y_train.shape[0], 1)
print(y_train.shape)
# Utilisation de np.unique avec return_counts=True pour obtenir les valeurs uniques et leurs occurrences dans y_train
print(np.unique(y_train, return_counts=True))
#print(y_train == 0, y_train == 1)

# Affiche les dimensions des données de test et des étiquettes
print(x_test.shape)
y_test = y_test.reshape(y_test.shape[0], 1)
print(y_test.shape)
# Utilisation de np.unique avec return_counts=True pour obtenir les valeurs uniques et leurs occurrences dans y_test
print(np.unique(y_test, return_counts=True))
#print(y_test == 0, y_test == 1)

# Création d'une figure
plt.figure(figsize=(16, 8))

# Une boucle sur les 9 premières images et leurs étiquettes pour les afficher
for i in range(1, 10):
    plt.subplot(4, 5, i)  # sous graphique
    plt.imshow(x_train[i], cmap='gray')  # Affiche l'image (x_train[i])
    plt.title(y_train[i])  # Ajoute le titre du sous graphique en affichant les étiquettes
    plt.tight_layout()  # Ajustement
plt.show()

y_train = y_train.T
y_test = y_test.T

# Réorganise les données d'entrainement x_train pour qu'elles soient de forme (nombre d'échantillons, -1)
# et divise toutes les valeurs par la valeur maximale de x_train pour normaliser les données.
x_train = x_train.T
x_train_reshape = x_train.reshape(-1, x_train.shape[-1]) / x_train.max()

# Calcule la valeur maximale dans les données d'entrainement x_train_reshape après la transformation et la normalisation.
x_train_reshape.max()

# Réorganise les données d'entrainement x_test pour qu'elles soient de forme (nombre d'échantillons, -1)
# et divise toutes les valeurs par la valeur maximale de x_train pour normaliser les données.
x_test = x_test.T
x_test_reshape = x_test.reshape(-1, x_test.shape[-1]) / x_train.max()

# Calcule la valeur maximale dans les données d'entrainement x_test_reshape après la transformation et la normalisation.
x_test_reshape.max()

# m_train = 300
# m_test = 80
# x_test_reshape = x_test_reshape[:, :m_test]
# x_train_reshape = x_train_reshape[:, :m_train]
# y_train = y_train[:, :m_train]
# y_test = y_test[:, :m_test]

# Affiche la dimension des données d'entraînement x_train_reshape après la transformation et la normalisation.
print(x_train_reshape.shape)
# Affiche la dimension des données d'entraînement x_test_reshape après la transformation et la normalisation.
print(x_test_reshape.shape)

print(y_train.shape)
print(y_test.shape)

# Avant l'initialisation des paramètres
print("Dimensions de x_train:", x_train.shape)
print("Dimensions de y_train:", y_train.shape)

# Initialisation des paramètres du modèle
def initialisation(n0, n1, n2):
    W1 = np.random.randn(n1, n0)
    b1 = np.random.randn(n1, 1)
    W2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    print("Dimensions de W1:", W1.shape)
    print("Dimensions de b1:", b1.shape)
    print("Dimensions de W2:", W2.shape)
    print("Dimensions de b2:", b2.shape)

    return parametres

#Calcule la sortie d'un modèle basé sur une combinaison linéaire des caractéristiques d'entrée X avec des poids W et un biais b, suivi de l'application de la fonction sigmoïde.
def forward_propagation(X, parametres):
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    Z1 = W1.dot(X) + b1
    print("Dimensions de Z1 avant activation:", Z1.shape)
    A1 = 1 / (1 + np.exp(-Z1))
    print("Dimensions de A1 après activation:", A1.shape)
    Z2 = W2.dot(A1) + b2
    print("Dimensions de Z2 avant activation:", Z2.shape)
    A2 = 1 / (1 + np.exp(-Z2))
    print("Dimensions de A2 après activation:", A2.shape)

    activations = {
        'A1': A1,
        'A2': A2
    }

    return activations

# Régression logisitique
# Calcul du coût (la perte)
def logLoss(A, y):  # Ai est la probabilité prédite par le modèle pour i
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

# Gradients
# Calcul les gradients de la perte par rapport aux paramètres ( poids et vecteur)
def back_propagation(X, y, activations, parametres):
    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']

    m = y.shape[1]

    dZ2 = A2 - y
    print("Dimensions de dZ2:", dZ2.shape)
    dW2 = 1 / m * dZ2.dot(A1.T)
    print("Dimensions de dW2:", dW2.shape)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    print("Dimensions de dZ1:", dZ1.shape)
    dW1 = 1 / m * dZ1.dot(X.T)
    print("Dimensions de dW2:", dW1.shape)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return gradients

# Mettre à jour les paramètres (poids et vecteur) en utilisant les gradients utilisés
def update(gradients, parametres, learning_rate):
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    #print(A)
    return parametres

# Faire les prédictions à l'aide d'un modèle Rlogi
def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    #print(activations)
    A2 = activations['A2']
    #print(A2)

    return A2 >= 0.5

# Entraîne un modèle de régression logistique (neurone artificiel) en utilisant la descente de gradient stochastique.
def neural_network(x_train, y_train, x_test, y_test, n1 = 32, learning_rate=0.01, n_iter=8000):
    #initialisation W, b
    print("Dans neural_network :", y_train.shape)
    print("Dans neural_network :", x_train.shape)
    n0 = x_train.shape[0]
    n2 = x_train.shape[0]
    parametres = initialisation(n0, n1, n2)

    print("Dans neural_network :", parametres)
    train_cout = []
    train_accuracy = []
    test_cout = []
    test_accuracy = []
    cout = []
    accuracy = []

    for i in tqdm(range(n_iter)):

        #mise à jour
        activations = forward_propagation(x_train, parametres)
        gradients = back_propagation(x_train, y_train, activations, parametres)
        parametres = update(gradients, parametres, learning_rate)

        if i % 10 == 0:
            #train cout/accuracy
            train_cout.append(logLoss(y_train, activations['A2']))
            #precision
            y_pred = predict(x_train, parametres)
            print("Dans neural_network après y_pred:", y_train.shape)
            print("Dans neural_network après y_pred:", x_train.shape)
            print("Dans neural_network après y_pred:", y_pred.shape)
            print(y_pred.flatten())

            current_accuracy = accuracy_score(y_train.flatten(), y_pred.flatten())
            train_accuracy.append(current_accuracy)

            #test cout/accuracy
            A_test = forward_propagation(x_test, parametres)
            test_cout.append(logLoss(y_test, A_test))

            #precision
            y_pred = predict(x_test, parametres)

            current_accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
            test_accuracy.append(current_accuracy)

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_cout, label='train_cout')
    plt.plot(test_cout, label='test_cout')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='train_accuracy')
    plt.plot(test_accuracy, label='test_accuracy')
    plt.legend()
    plt.show()
    return parametres

# Appel de la fonction et affichage des courbes de coût et d'exactitude pour l'entraînement et le test
parametres1 = neural_network(x_train_reshape, y_train, x_test_reshape, y_test)