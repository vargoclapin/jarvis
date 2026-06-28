from asyncio import subprocess
import platform
import pygame
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os


import cv2
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


WIDTH, HEIGHT = 1000, 700
FPS = 60
FOV = 500

BLACK = np.array([10, 10, 15])
WHITE = np.array([255, 255, 255])
FOLDER_COLOR = np.array([100, 200, 255]) # Bleu fluo pour les dossiers
FILE_COLOR = np.array([200, 255, 100])   # Vert fluo pour les fichiers/sous-dossiers
LINE_COLOR = np.array([50, 80, 120])
COLOR_MOUSE = np.array([255, 100, 100]).astype(np.uint8)  # Rouge fluo pour le pointeur de la souris


class Folder3D():
    def __init__(self, name, x, y, z, children=None, is_open=False, file_path=None):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.file_path = file_path
        self.is_open = is_open
        self.children = children
    
    def draw_folder(self, pixel_buffer, camera, labels_to_draw, cam_trig):
        # Calculer la position projetée du dossier
        projected_x, projected_y = project_3d_to_2d(self.x, self.y, self.z, camera, cam_trig)
        
        if projected_x is None:
            # Si hors écran/derrière la caméra, on retourne quand même le format attendu (Buffer, X, Y)
            return pixel_buffer, None, None
        
        draw_star(pixel_buffer, int(projected_x), int(projected_y), 50, FOLDER_COLOR, star_cache=STAR_CACHE)
        labels_to_draw.append((self.name, int(projected_x) + 20, int(projected_y) - 10))
        
        # Dessiner les lignes vers les enfants
        if self.children is None:
            return pixel_buffer, projected_x, projected_y
            
        for child in self.children:
            if child.is_open:
                # OPTIMISATION : L'enfant calcule sa position et nous la renvoie (plus besoin de la calculer nous-mêmes)
                pixel_buffer, child_px, child_py = child.draw_folder(pixel_buffer, camera, labels_to_draw, cam_trig)
                
                if child_px is not None and child_py is not None:
                    draw_line(pixel_buffer, int(projected_x), int(projected_y), int(child_px), int(child_py), LINE_COLOR)
        
        # On retourne le buffer mis à jour ET les coordonnées de ce dossier
        return pixel_buffer, projected_x, projected_y
    
    def find_nearest_in_children(self, camera, cam_trig, cx, cy):
        nearest_folder = None
        min_distance_sq = float('inf') # OPTIMISATION : On compare les carrés des distances
        
        if self.children is not None:
            for child in self.children:
                if child.is_open:
                    x, y = project_3d_to_2d(child.x, child.y, child.z, camera, cam_trig)
                    if x is not None and y is not None:
                        # Plus de math.sqrt() !
                        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                        if dist_sq < min_distance_sq:
                            min_distance_sq = dist_sq
                            nearest_folder = child
                            
                    child_nearest, child_dist_sq = child.find_nearest_in_children(camera, cam_trig, cx, cy)
                    if child_dist_sq < min_distance_sq:
                        min_distance_sq = child_dist_sq
                        nearest_folder = child_nearest
        
        return nearest_folder, min_distance_sq
    
    def ouvrir_sur_ordinateur(self):
        """Ouvre le fichier ou le dossier dans l'explorateur réel de l'OS"""
        if self.file_path and os.path.exists(self.file_path):
            try:
                if platform.system() == 'Windows':
                    os.startfile(self.file_path)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.call(('open', self.file_path))
                print(f"Ouverture de : {self.file_path}")
                return True
            except Exception as e:
                print(f"Erreur lors de l'ouverture : {e}")
        else:
            print("Chemin invalide ou inexistant.")
        return False

class Bureau3D():
    def __init__(self, folders):
        self.folders = folders

    def find_nearest_folder(self, camera, cam_trig):
        nearest_folder = None
        min_distance_sq = float('inf')
        
        # OPTIMISATION : On ne calcule le centre de l'écran qu'une seule fois
        cx = WIDTH // 2
        cy = HEIGHT // 2
        
        for folder in self.folders:
            if folder.is_open:
                x, y = project_3d_to_2d(folder.x, folder.y, folder.z, camera, cam_trig)
                if x is not None and y is not None:
                    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                    if dist_sq < min_distance_sq:
                        min_distance_sq = dist_sq
                        nearest_folder = folder
                        
                child_nearest, child_dist_sq = folder.find_nearest_in_children(camera, cam_trig, cx, cy)
                if child_dist_sq < min_distance_sq:
                    min_distance_sq = child_dist_sq
                    nearest_folder = child_nearest
        
        # On ne fait la vraie racine carrée lourde qu'à la toute fin si on en a vraiment besoin
        actual_min_distance = math.sqrt(min_distance_sq) if min_distance_sq != float('inf') else float('inf')
        return nearest_folder, actual_min_distance

class Camera():
    def __init__(self, x, y, z, angle_x=0, angle_y=0):
        self.x = x
        self.y = y
        self.z = z
        self.angle_x = angle_x
        self.angle_y = angle_y




def draw_line(pixel_buffer, x1, y1, x2, y2, color):
    # Récupérer les dimensions du buffer (si WIDTH et HEIGHT ne sont pas globaux)
    HEIGHT, WIDTH = pixel_buffer.shape[:2]
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    # 1. Précalcul du dégradé (Halo) pour une rapidité extrême
    # On simule la courbe exponentielle de ton étoile sans faire de maths dans la boucle
    c_center = color
    c_halo_1 = color * 0.50  # halo proche
    c_halo_2 = color * 0.15  # halo lointain

    # 2. Déterminer l'axe majeur pour tracer le halo perpendiculairement
    is_x_major = dx > dy

    while True:
        # Tracer le cœur très brillant de la ligne
        if 0 <= x1 < WIDTH and 0 <= y1 < HEIGHT:
            pixel_buffer[y1, x1] = c_center
            
        # 3. Tracer le halo perpendiculaire pour créer l'effet de diffusion
        if is_x_major:
            # La ligne avance surtout à l'horizontale -> on fait un halo vertical
            if 0 <= x1 < WIDTH:
                if 0 <= y1 - 1 < HEIGHT: pixel_buffer[y1 - 1, x1] = c_halo_1
                if 0 <= y1 + 1 < HEIGHT: pixel_buffer[y1 + 1, x1] = c_halo_1
                if 0 <= y1 - 2 < HEIGHT: pixel_buffer[y1 - 2, x1] = c_halo_2
                if 0 <= y1 + 2 < HEIGHT: pixel_buffer[y1 + 2, x1] = c_halo_2
        else:
            # La ligne avance surtout à la verticale -> on fait un halo horizontal
            if 0 <= y1 < HEIGHT:
                if 0 <= x1 - 1 < WIDTH: pixel_buffer[y1, x1 - 1] = c_halo_1
                if 0 <= x1 + 1 < WIDTH: pixel_buffer[y1, x1 + 1] = c_halo_1
                if 0 <= x1 - 2 < WIDTH: pixel_buffer[y1, x1 - 2] = c_halo_2
                if 0 <= x1 + 2 < WIDTH: pixel_buffer[y1, x1 + 2] = c_halo_2

        # Algorithme classique de Bresenham pour avancer
        if x1 == x2 and y1 == y2:
            break
        err2 = err * 2
        if err2 > -dy:
            err -= dy
            x1 += sx
        if err2 < dx:
            err += dx
            y1 += sy

            
STAR_CACHE = {}

def get_star_kernel(size, star_cache=STAR_CACHE):
    """Calcule la matrice de l'étoile une seule fois par taille et la sauvegarde."""
    if size not in star_cache:
        # On utilise la "vectorisation" NumPy pour calculer toute la grille d'un coup
        # C'est 100x plus rapide qu'une double boucle for
        grid1d = np.arange(-size, size + 1)
        I, J = np.meshgrid(grid1d, grid1d)
        
        # On stocke uniquement le coefficient d'intensité (entre 0 et 1)
        star_cache[size] = np.exp(-((I**2 + J**2) / (4 * size)))
        
    return star_cache[size]

def draw_star(pixel_buffer_float, x, y, size, color_array, star_cache=STAR_CACHE):
    """
    OPTIMISATION : 
    - pixel_buffer_float DOIT être un tableau np.float32 ou float64 (et non uint8).
    - color_array DOIT être un np.array([R, G, B]).
    - Il n'y a plus de clip() ou de astype() ici, on additionne purement et simplement.
    """
    HEIGHT, WIDTH = pixel_buffer_float.shape[:2]
    
    kernel = get_star_kernel(size, star_cache)
    
    x_min = max(0, x - size)
    x_max = min(WIDTH, x + size + 1)
    y_min = max(0, y - size)
    y_max = min(HEIGHT, y + size + 1)
    
    if x_min >= WIDTH or x_max <= 0 or y_min >= HEIGHT or y_max <= 0:
        return

    k_x_min = x_min - (x - size)
    k_x_max = k_x_min + (x_max - x_min)
    k_y_min = y_min - (y - size)
    k_y_max = k_y_min + (y_max - y_min)

    patch = kernel[k_y_min:k_y_max, k_x_min:k_x_max]
    
    # OPTIMISATION : Plus de np.clip, plus de astype(uint8), plus de isinstance.
    # On ajoute juste les valeurs flottantes directement. Le '...' gère la diffusion (broadcasting) RGB.
    pixel_buffer_float[y_min:y_max, x_min:x_max] += color_array * patch[..., np.newaxis]


def project_3d_to_2d(x, y, z, camera, cam_trig):
    """
    OPTIMISATION :
    On passe un dictionnaire (ou objet) 'cam_trig' qui contient les cos/sin 
    déjà calculés pour cette frame.
    """
    dx = x - camera.x
    dy = y - camera.y
    dz = z - camera.z

    # Utilisation des valeurs précalculées
    dx_rot = dx * cam_trig['cos_y'] + dz * cam_trig['sin_y']
    dz_rot = -dx * cam_trig['sin_y'] + dz * cam_trig['cos_y']

    dy_rot = dy * cam_trig['cos_x'] - dz_rot * cam_trig['sin_x']
    dz_final = dy * cam_trig['sin_x'] + dz_rot * cam_trig['cos_x']

    if dz_final <= 0.1:
        return None, None

    proj_x = dx_rot * (FOV / dz_final) + WIDTH // 2
    proj_y = dy_rot * (FOV / dz_final) + HEIGHT // 2

    return int(proj_x), int(proj_y)

def generate_random_structure(number_of_folders, x_min= -500, x_max=500, y_min= -500, y_max=500, z_min=0, z_max=500):
    positions = []
    for _ in range(number_of_folders):
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        z = random.randint(z_min, z_max)
        positions.append((x, y, z))
    
    root_folders = [Folder3D(f"Folder_{i}", x, y, z, None, [], is_open=True) for i, (x, y, z) in enumerate(positions)]
    return Bureau3D(root_folders)

def draw_pointer(pixel_buffer):
    w, l = pixel_buffer.shape[1], pixel_buffer.shape[0]
    mouse_x, mouse_y = w//2, l//2
    for i in range (-3, 4):
        for j in range (-10, 11):
            pixel_buffer[mouse_y + i, mouse_x + j] = COLOR_MOUSE
    for i in range (-10, 11):
        for j in range (-3, 4):
            pixel_buffer[mouse_y + i, mouse_x + j] = COLOR_MOUSE
    return pixel_buffer

def generate_sphere_repartition(centre, rayon, N):
    theta_or = math.pi * (3 - math.sqrt(5))
    rep = []  
    if N == 0:
        return []
    if N == 1:
        return [centre]
    for i in range(N):
        z = (1 - (i / float(N - 1)) * 2)  # z goes from 1 to -1
        x = math.sqrt(1-z**2)*math.cos(theta_or * i)
        y = math.sqrt(1-z**2)*math.sin(theta_or * i)
        rep.append((rayon*x + centre[0], rayon*y + centre[1], rayon*z + centre[2]))
    return rep

def generate_spheric_bureau(list_children, centre, rayon, parent=None):
    if len(list_children) == 0:
        return Bureau3D([])
    N = list_children[0]
    bureau = []
    rep = generate_sphere_repartition(centre, rayon, N) 
    
    for i in range(N):
        x, y, z = rep[i]
        
        folder = Folder3D(f"Folder_{i}", x, y, z, parent, [], is_open=False)
        
        if len(list_children) > 1:
            sous_bureau = generate_spheric_bureau(list_children[1:], (x, y, z), rayon /3)
            folder.children = sous_bureau.folders
            
        bureau.append(folder)
        
    return Bureau3D(bureau)


def generer_depuis_mon_bureau(chemin_dossier, centre, rayon, profondeur_max=2, profondeur_act=0):
    if not os.path.exists(chemin_dossier):
        print(f"[Alerte] Le chemin n'existe pas : {chemin_dossier}")
        return Bureau3D([])
        
    if profondeur_act > profondeur_max:
        return Bureau3D([])
        
    bureau = []
    try:
        if os.path.isdir(chemin_dossier):
            contenu = os.listdir(chemin_dossier)
            N = len(contenu)
            
            # Message de succès pour la racine
            if profondeur_act == 0:
                print(f"[Génération] Lecture de la racine OK : {N} éléments trouvés.")
                
            if N == 0:
                return Bureau3D([])
                
            rep = generate_sphere_repartition(centre, rayon, N)
            
            for i, nom_fichier in enumerate(contenu):
                x, y, z = rep[i]
                chemin_complet = os.path.join(chemin_dossier, nom_fichier)
                
                # --- LA CORRECTION EST ICI ---
                # Tout est nommé explicitement pour éviter la confusion des paramètres
                folder = Folder3D(
                    name=nom_fichier, 
                    x=x, 
                    y=y, 
                    z=z,  
                    children=[], 
                    is_open=False, 
                    file_path=chemin_complet
                )
                
                if os.path.isdir(chemin_complet):
                    sous_bureau = generer_depuis_mon_bureau(
                        chemin_complet, 
                        (x, y, z), 
                        rayon / 3,  
                        profondeur_max=profondeur_max,
                        profondeur_act=profondeur_act + 1
                    )
                    folder.children = sous_bureau.folders
                    
                bureau.append(folder)
                
        return Bureau3D(bureau)

    except Exception as e:
        if profondeur_act == 0:
            print(f"[ERREUR FATALE À LA RACINE] : {e}")
        else:
            print(f"[Erreur sur un sous-dossier ignoré] {chemin_dossier} -> {e}")
            
        return Bureau3D([])

def norme(a,b):
    return ((a.x-b.x)**2+(a.y-b.y)**2)**0.5 

def calcul_barycentre(landmarks):
    centre=[0,0]
    n=len(landmarks)
    for p in landmarks:
        centre[0]+=p.x/n
        centre[1]+=p.y/n
    return centre
    


def doigt_est_plie(tip, pip, wrist):
    
    dist_tip_poignet = norme(tip, wrist)
    dist_pip_poignet = norme(pip, wrist)
    
    
    return dist_tip_poignet < dist_pip_poignet

def est_poing_ferme(landmarks):
    poignet = landmarks[0]
    
    # On vérifie les 4 doigts principaux (on exclut souvent le pouce 
    # car sa mécanique de pliure est différente)
    index_plie = doigt_est_plie(landmarks[8], landmarks[6], poignet)
    majeur_plie = doigt_est_plie(landmarks[12], landmarks[10], poignet)
    annulaire_plie = doigt_est_plie(landmarks[16], landmarks[14], poignet)
    auriculaire_plie = doigt_est_plie(landmarks[20], landmarks[18], poignet)
    
    # Si les 4 doigts sont pliés vers le poignet, c'est un poing !
    if index_plie and majeur_plie and annulaire_plie and auriculaire_plie:
        return True
    return False

def est_corne(landmarks):
    poignet = landmarks[0]
    
    index_plie = doigt_est_plie(landmarks[8], landmarks[6], poignet)
    majeur_plie = doigt_est_plie(landmarks[12], landmarks[10], poignet)
    annulaire_plie = doigt_est_plie(landmarks[16], landmarks[14], poignet)
    auriculaire_plie = doigt_est_plie(landmarks[20], landmarks[18], poignet)
    if not index_plie and majeur_plie and annulaire_plie and not auriculaire_plie:
        return True
    else : 
        return False
    
def deep_three(landmarks):
    poignet = landmarks[0]

    majeur_plie = doigt_est_plie(landmarks[12], landmarks[10], poignet)
    annulaire_plie = doigt_est_plie(landmarks[16], landmarks[14], poignet)
    auriculaire_plie = doigt_est_plie(landmarks[20], landmarks[18], poignet)

    d= norme(landmarks[4],landmarks[8])<0.1
    if d and not majeur_plie and not annulaire_plie and not auriculaire_plie:
        return True
    return False


def zoom(landmarks):
    aire = norme(landmarks[17],landmarks[5])*norme(landmarks[5],landmarks[0])
    min = 0.006
    max = 0.09
    inf = 0.009
    sup =0.02
    if inf<=aire <=sup :
        vitesse = 0
    elif sup<aire<=max:
        vitesse = 7/(max-sup)*(aire-sup)+1
    elif max<aire:
        vitesse = 8
    elif aire<min:
        vitesse=-8
    else :
        vitesse = 7/(inf-min)*(aire-min)-8
    return vitesse

def tourni(landmarks): #ici on renvoie la vitesse angualaire de la caméra en fonction de la position de la main (on a une vitesse pour x et y)
    centre = calcul_barycentre(landmarks)
    en_y = 0.06*round(centre[0]*50)/50 -0.03
    en_x = 0.06*round(centre[1]*50)/50 -0.03
    return (round(en_y,3),round(en_x,3))




