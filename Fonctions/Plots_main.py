import numpy as np
import matplotlib.pyplot as plt
import nbimporter
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint


# Fonction pour comparer les différents modèles de débruitage
def plot_comparatif(X_test, Y_test, img_lisse, i_fc, titre, normalization_factor):

    vmin = np.min([np.min(X_test[i_fc]), np.min(img_lisse), np.min(Y_test[i_fc])]) # Cela pour avoir les mêmes échelles sur tous les plots : ce choix lisse hélas ls variations de l'image cible...
    vmax = np.max([np.max(X_test[i_fc]), np.max(img_lisse), np.max(Y_test[i_fc])]) # ... mais permet la comparaison des trois images entre elles.

    sh = 0.7 # réduire la taille de la colorbar
    cmp = 'coolwarm'
    plt.figure(figsize=(10, 4))

    # Plot image originale
    plt.subplot(1, 3, 1)
    im = plt.imshow(X_test[i_fc]/normalization_factor, cmap=cmp, vmin=vmin, vmax=vmax) # en faisat "/ normalization_factor", on affiche les valeurs réelles de l'image, ont les dénormalise
    plt.title('Image originale', fontweight = 'bold')
    #plt.colorbar(im, ax=plt.gca(),shrink = sh)

    # Plot image lissée
    plt.subplot(1, 3, 2)
    im = plt.imshow(img_lisse/normalization_factor, cmap=cmp, vmin=vmin, vmax=vmax)
    plt.title('Image lissée', fontweight = 'bold')
    #plt.colorbar(im, ax=plt.gca(),shrink = sh)

    # Plot image cible + colorbar pour les 3 (car même échelle !!!)
    plt.subplot(1, 3, 3)
    im = plt.imshow(Y_test[i_fc]/normalization_factor, cmap=cmp, vmin=vmin, vmax=vmax)
    plt.title('Image cible', fontweight = 'bold')
    
    cbar = plt.colorbar(im, ax=plt.gcf().get_axes(), shrink=sh, label='Sea Surface Height', aspect=10, pad=-0.44)
    cbar.ax.yaxis.set_label_coords(4.6, 0.5)  # Position du label de la colorbar
    cbar.ax.set_position([cbar.ax.get_position().x0, cbar.ax.get_position().y0 - 0.035,
                          cbar.ax.get_position().width, cbar.ax.get_position().height])

    cbar.set_label('Sea Surface Height', labelpad=15, rotation=270, fontweight='bold')

    plt.suptitle(titre, fontweight = 'bold')
    plt.tight_layout()
    
    plt.show()



def train_and_plot(X_train, Y_train, longueur_img, largeur_img, num_conv_layers, colors, epoch=10, batch_sz=80):
    model = build_model(longueur_img, largeur_img, num_conv_layers)
    
    # Point de contrôle pour sauvegarder le meilleur modèle basé sur la perte de validation
    checkpoint = ModelCheckpoint(f'./Saves/Models/model_{num_conv_layers}_conv_layers.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=0)
    
    history = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_sz, verbose=0, validation_split=0.2, callbacks=[checkpoint])

    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    plt.plot(history.history['loss'], label=f'Train loss ({num_conv_layers+1} couches)',color = colors,linestyle = '-')
    plt.plot(history.history['val_loss'], label=f'Validation loss ({num_conv_layers+1} couches)',color = colors,linestyle = '--')
    plt.xlabel('Itérations',fontweight='bold')
    plt.ylabel('Loss',fontweight='bold')
    plt.title(f"Entraînement d'un modèle avec de 1 à {num_conv_layers+1} couches convolutives")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Charger le meilleur modèle sauvegardé
    best_model = keras.models.load_model(f'./Saves/Models/model_{num_conv_layers}_conv_layers.h5')

    return train_loss, val_loss, best_model



def train_and_plot_2(X_train, Y_train, longueur_img, largeur_img, num_conv_layers, colors, epoch=10, batch_sz=30):
    model = build_model(longueur_img, largeur_img, num_conv_layers)
    
    # Point de contrôle pour sauvegarder le meilleur modèle basé sur la perte de validation
    checkpoint = ModelCheckpoint(f'./Saves/Models/model_{num_conv_layers}_{batch_sz}_conv_layers.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=0)
    
    history = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_sz, verbose=0, validation_split=0.2, callbacks=[checkpoint])

    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    
    plt.plot(history.history['loss'], label=f'Train loss (batch size ={batch_sz})',color = colors,linestyle = '-')
    plt.plot(history.history['val_loss'], label=f'Validation loss (batch size ={batch_sz})',color = colors,linestyle = '--')
    plt.xlabel('Itérations',fontweight='bold')
    plt.ylabel('Loss',fontweight='bold')
    plt.title(f"Entraînement d'un modèle selon batch size")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Charger le meilleur modèle sauvegardé
    best_model = keras.models.load_model(f'./Saves/Models/model_{num_conv_layers}_{batch_sz}_conv_layers.h5')

    return train_loss, val_loss, best_model

def train_and_plot_3(X_train, Y_train, longueur_img, largeur_img, num_conv_layers, colors, filter_i, epoch=10, batch_sz=30):
    model = build_model(longueur_img, largeur_img, num_conv_layers, filter = filter_i)
    
    # Point de contrôle pour sauvegarder le meilleur modèle basé sur la perte de validation
    checkpoint = ModelCheckpoint(f'./Saves/Models/model_{num_conv_layers}_{batch_sz}_{filter_i}_conv_layers.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=0)
    
    history = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_sz, verbose=0, validation_split=0.2, callbacks=[checkpoint])

    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    
    plt.plot(history.history['loss'], label=f'Train loss (filtre conv ={filter_i}x{filter_i})',color = colors,linestyle = '-')
    plt.plot(history.history['val_loss'], label=f'Validation loss (filtre conv ={filter_i}x{filter_i})',color = colors,linestyle = '--')
    plt.xlabel('Itérations',fontweight='bold')
    plt.ylabel('Loss',fontweight='bold')
    plt.title(f"Entraînement d'un modèle selon le filtre convolutif")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Charger le meilleur modèle sauvegardé
    best_model = keras.models.load_model(f'./Saves/Models/model_{num_conv_layers}_{batch_sz}_{filter_i}_conv_layers.h5')

    return train_loss, val_loss, best_model


def train_and_plot_4(X_train, Y_train, longueur_img, largeur_img, num_conv_layers, colors, filter_i, active_func, epoch=10, batch_sz=30):
    model = build_model(longueur_img, largeur_img, num_conv_layers, filter = filter_i, activ_fc = active_func)
    
    # Point de contrôle pour sauvegarder le meilleur modèle basé sur la perte de validation
    checkpoint = ModelCheckpoint(f'./Saves/Models/model_{num_conv_layers}_{batch_sz}_{filter_i}_{active_func}_conv_layers.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=0)
    
    history = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_sz, verbose=0, validation_split=0.2, callbacks=[checkpoint])

    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    
    plt.plot(history.history['loss'], label=f"Train loss (fonction d'activation ={active_func})",color = colors,linestyle = '-')
    plt.plot(history.history['val_loss'], label=f"Validation loss (fonction d'activation ={active_func})",color = colors,linestyle = '--')
    plt.xlabel('Itérations',fontweight='bold')
    plt.ylabel('Loss',fontweight='bold')
    plt.title(f"Entraînement d'un modèle selon la fonction d'activation")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Charger le meilleur modèle sauvegardé
    best_model = keras.models.load_model(f'./Saves/Models/model_{num_conv_layers}_{batch_sz}_{filter_i}_{active_func}_conv_layers.h5')

    return train_loss, val_loss, best_model


def train_and_plot_5(X_train, Y_train, longueur_img, largeur_img, num_conv_layers, colors, filter_i, active_func, add_choix,epoch=10, batch_sz=30):
    model = build_model(longueur_img, largeur_img, num_conv_layers, filter = filter_i, activ_fc = active_func, add = add_choix)
    
    # Point de contrôle pour sauvegarder le meilleur modèle basé sur la perte de validation
    checkpoint = ModelCheckpoint(f'./Saves/Models/model_{num_conv_layers}_{batch_sz}_{filter_i}_{active_func}_Pooling_{add_choix}_conv_layers.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=0)
    
    history = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_sz, verbose=0, validation_split=0.2, callbacks=[checkpoint])

    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    # Cette fois on n'a pas besoin de faire le plot habituel.
    
    # Charger le meilleur modèle sauvegardé
    best_model = keras.models.load_model(f'./Saves/Models/model_{num_conv_layers}_{batch_sz}_{filter_i}_{active_func}_Pooling_{add_choix}_conv_layers.h5')

    return train_loss, val_loss, best_model
    

# On fournit la fonction buld_model pour les fonctions "train_and_plot"
def build_model(long, larg, num_conv_layers, dropout_rate=0.25, filter = 2, activ_fc = 'relu', add=False):
    model = keras.models.Sequential()                    # On crée le modèle
    model.add(keras.layers.Input(shape=(long, larg, 1))) # Couche d'input

    for _ in range(num_conv_layers):  # Selon le nombre demandé, on ajoute des couches de convolution et de dropout
        if add == True:               # Si add est True, on va ajouter du MaxPooling en plus des convolutions et dropout (voir partie 3.5)
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(32, (filter, filter), activation=activ_fc, padding='same')) # La couche de convolution
        if add == True:
            model.add(keras.layers.UpSampling2D(size=(2, 2))) # Réaugmenter l'image en cas de Max Pooling
        model.add(keras.layers.Dropout(dropout_rate))         # Couche de dropout : elle permet d'éteindre un pourcentage du réseau de neurones à chaque itération

    model.add(keras.layers.Conv2D(1, (filter, filter), activation=None, padding='same'))       # Couche finale

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error']) # Compilation du modèle
    return model
