import pickle

# Fonction pour sauvegarder les données
def sauvegarder_donnees(cle, donnees,XY):
    fichier = f"./Saves/26x{cle}_{XY}.pkl"
    with open(fichier, 'wb') as f:
        pickle.dump(donnees, f)
    print(f"Données sauvegardées avec la clé {cle} dans {fichier}")

# Fonction pour réouvrir les données
def ouvrir_donnees(cle,XY):
    fichier = f"./Saves/26x{cle}_{XY}.pkl"
    with open(fichier, 'rb') as f:
        donnees = pickle.load(f)
    return donnees