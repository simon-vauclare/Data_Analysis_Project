import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from src.swot import *
import glob

# Fonction pour présenter les données
def show_data(i):
    # Figure + axe avec projection cartographique
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.COASTLINE) # Côtes des continents
    ax.add_feature(cfeature.BORDERS, linestyle=':') # Frontières des pays
    ax.add_feature(cfeature.LAND, edgecolor='black') # Terres continentales
    
    # Tracer les données avec pcolormesh
    ds = SwotTrack(sorted(glob.glob('./Inputs/input_ssh_karin_013_*.nc'))[i])._dset
    c = ax.pcolormesh(ds.longitude, ds.latitude, ds.ssh_true, cmap='viridis', transform=ccrs.PlateCarree())
    
    bar = fig.colorbar(c, ax=ax, label='SSH True')
    plt.title('Carte de la hauteur de la mer')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('./Saves/Figures/1_show_data.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('"1_show_data.png" Sauvegardé !')

# Fonction pour présenter le découpage des données - 1ère partie
def show_cutting_1(i):
    ds = SwotTrack(sorted(glob.glob('./Inputs/input_ssh_karin_013_*.nc'))[i])._dset
    
    ds['num_lines'] = 2*(ds['num_lines']-ds['num_lines'][0]) # en kilomètres
    ds['num_pixels'] = 2*ds['num_pixels'] # en kilomètres
    
    # Montrer le découpage en deux longueurs de 52km
    
    def labels(i):
        ax[i].set_xlabel('km',fontweight='bold')
        ax[i].set_ylabel('km',fontweight='bold')
    
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    
    ds.ssh_true.plot(ax=ax[0])
    ax[0].set_title('Avant le tri : ssh_true',y=1.05,fontweight='bold')
    labels(0)
    
    ds_line1 = ds.where((ds.num_pixels >= 10) & (ds.num_pixels <= 60),drop=True) 
    ds_line1.ssh_true.plot(ax=ax[1])
    ax[1].set_title('Après le tri : ssh_true, ligne 1',y=1.05,fontweight='bold')
    labels(1)
    
    ds_line2 = ds.where((ds.num_pixels >= 80) & (ds.num_pixels <= 130),drop=True) 
    ds_line2.ssh_true.plot(ax=ax[2])
    ax[2].set_title('Après le tri : ssh_true, ligne 2',y=1.05,fontweight='bold')
    labels(2)
    
    plt.tight_layout()
    plt.savefig('./Saves/Figures/2_1_show_cutting.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('"2_1_show_cutting.png" Sauvegardé !')

# Fonction pour présenter le découpage des données - 2ème partie
def show_cutting_2(i,longeur_image):

    def labels(i,title):
        ax[i].set_xlabel('km',fontweight='bold')
        ax[i].set_ylabel('km',fontweight='bold')
        ax[i].set_title(title,y=1.03,fontweight='bold')
    
    ds = SwotTrack(sorted(glob.glob('./Inputs/input_ssh_karin_013_*.nc'))[i])._dset
    
    ds['num_lines'] = 2*(ds['num_lines']-ds['num_lines'][0]) # en kilomètres
    ds['num_pixels'] = 2*ds['num_pixels'] # en kilomètres
    
    ds_line = ds.where((ds.num_pixels >= 80) & (ds.num_pixels <= 130),drop=True) 
    
    shift = 13 # pour passer directement à la 10ème image
    
    min = 0 + shift * longeur_image
    max = min + longeur_image
    n_boxes = ds_line.num_lines[-1].values//longeur_image # nombre de bandes à faire en longueur
    
    keep_count = 0
    
    for j in range(2): # On n'affiche que les 5 premiers découpages
        ds_new_line = ds_line.where((ds_line.num_lines >= min) & (ds_line.num_lines < max),drop=True)
    
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
        if np.sum(np.isnan(ds_new_line.ssh_true)).values==0:
            plt.suptitle(f'Image {shift} : conservée (pas de NaN)',fontweight='bold')
        else:
            plt.suptitle(f'Image {shift} : non conservée (présence de NaN)',fontweight='bold')
        
        ds_new_line.ssh_karin.plot(ax=ax[0],cmap='viridis')
        labels(0,'KARIN')
        ds_new_line.ssh_true.plot(ax=ax[1],cmap='viridis')
        labels(1,'TRUE')
    
        min += longeur_image
        max += longeur_image
    
        shift +=1
        plt.savefig('./Saves/Figures/2_2_'+str(shift)+'_show_cutting.png', dpi=300, bbox_inches='tight')
        print('"2_2_'+str(shift)+'_show_cutting.png" Sauvegardé !')
        plt.close()