import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
#!pip install h5py
#from utilities import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import glob
import cv2

def images_to_hdf5_dataset(imageFormat, data_repo, h5file, img_width, img_height):
    nfiles = len(glob.glob(data_repo+"*.jpg"))
    print(f'count of image files nfiles={nfiles}')

    #on load les images de chats du training set pour les transformer en matrices
    with h5py.File(h5file,'w') as  h5f:
        x = h5f.create_dataset('X',shape=(nfiles, img_width, img_height), dtype=int)
        y = h5f.create_dataset('y',shape=(nfiles,), dtype=int)

        for cnt, ifile in tqdm(enumerate(glob.iglob(data_repo+"*."+imageFormat))):
            img = cv2.imread(ifile, cv2.IMREAD_GRAYSCALE)
            img_resize = cv2.resize( img, (img_width, img_height) )

            x[cnt] = img_resize
            #le label est 1 si la photo contient un chat
            if 'tulip' in ifile:
                y[cnt] = 1
            else:
                #le label est 0 si la photo contient un chien
                y[cnt] = 0

IMG_WIDTH = 64
IMG_HEIGHT = 64

repo_training_set = "../donnees_basique/train/"
repo_test_set = "../donnees_basique/test/"
dataset_train = "../datasets/train_set.hdf5"
dataset_test = "../datasets/test_set.hdf5"

#on convertit les images du train set en un fichier hdf5
images_to_hdf5_dataset("jpg", repo_training_set, dataset_train, IMG_WIDTH, IMG_HEIGHT)

#on fait la meme chose pour les images du test set
images_to_hdf5_dataset("jpg", repo_test_set, dataset_test, IMG_WIDTH, IMG_HEIGHT)