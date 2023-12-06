import glob
import h5py
import cv2
from tqdm import tqdm

def images_to_hdf5_dataset(imageFormat, data_repo, h5file, img_width, img_height):
    nfiles = len(glob.glob(data_repo+"*.jpg"))
    print(f'count of image files nfiles={nfiles}')

    #on load les images de chats du training set pour les transformer en matrices
    with h5py.File(h5file,'w') as h5f:
        x = h5f.create_dataset('X',shape=(nfiles, img_width, img_height), dtype=int)
        y = h5f.create_dataset('y',shape=(nfiles,), dtype=int)

        for cnt, ifile in tqdm(enumerate(glob.iglob(data_repo+"*."+imageFormat))):
            img = cv2.imread(ifile, cv2.IMREAD_GRAYSCALE)
            img_resize = cv2.resize( img, (img_width, img_height) )

            x[cnt] = img_resize
            #le label est 1 si la photo contient un chat
            if 'rose' in ifile:
                y[cnt] = 1
            else:
                #le label est 0 si la photo contient un tulip
                y[cnt] = 0

# Utilisation de la fonction
images_to_hdf5_dataset("jpg", "donnees_basique/train/", "trainset.hdf5", 64, 64)
#https://github.com/Momotoculteur/Image-classification/tree/master