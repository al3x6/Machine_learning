# # Nahid Emad November 23, 2023
#  Thanks to ChatGPT

import h5py
from PIL import Image
import numpy as np

def jpg_to_hdf5(input_jpg, output_hdf5):
    # Ouvrir l'image JPG
    image = Image.open(input_jpg)

    # Convertir l'image en tableau NumPy
    image_array = np.array(image)

    # Créer un fichier HDF5
    with h5py.File(output_hdf5, 'w') as hdf5_file:
        # Créer un jeu de données dans le fichier HDF5
        hdf5_file.create_dataset('image', data=image_array)

if __name__ == "__main__":
    # Spécifier le chemin de l'image JPG en entrée et le chemin du fichier HDF5 en sortie
    input_jpg_path = 'donnees_basique/train/1476943_ecf8d36b29_b.jpg'
    output_hdf5_path = 'oneflower.hdf5'

    # Convertir l'image JPG en fichier HDF5
    jpg_to_hdf5(input_jpg_path, output_hdf5_path)

    print(f"Conversion terminée. Fichier HDF5 créé à {output_hdf5_path}")
