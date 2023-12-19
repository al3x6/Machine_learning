import os

# Chemin du dossier contenant les images
dossier = "donnees_basique/train/tulip/"

# Liste tous les fichiers du dossier
fichiers = os.listdir(dossier)

# Compteur pour les nouveaux noms
compteur = 1

# Parcours des fichiers
for fichier in fichiers:
    # Vérifie si le fichier est un fichier JPG
    if fichier.endswith(".jpg") or fichier.endswith(".JPG"):
        # Crée le nouveau nom
        nouveau_nom = os.path.join(dossier, f"tulip.{compteur}.jpg")

        # Renomme le fichier
        ancien_chemin = os.path.join(dossier, fichier)
        os.rename(ancien_chemin, nouveau_nom)

        # Incrémente le compteur pour le prochain fichier
        compteur += 1