import os

def convertir_jpeg_en_jpg_dossier(dossier):
    # Vérifie si le dossier existe
    if not os.path.isdir(dossier):
        print("Le dossier spécifié n'existe pas.")
        return

    # Parcours tous les fichiers du dossier
    for nom_fichier in os.listdir(dossier):
        chemin_fichier = os.path.join(dossier, nom_fichier)

        # Vérifie si le chemin correspond à un fichier
        if os.path.isfile(chemin_fichier):
            # Vérifie si le fichier se termine par .jpeg
            if nom_fichier.lower().endswith('.jpeg'):
                nouveau_nom = os.path.join(dossier, nom_fichier[:-5] + '.jpg')
                # Renomme le fichier
                os.rename(chemin_fichier, nouveau_nom)
                print(f"Le fichier {nom_fichier} a été renommé en {nouveau_nom}")

# Utilisation de la fonction pour convertir les fichiers dans un dossier
convertir_jpeg_en_jpg_dossier('donnees_basique/rose/')
