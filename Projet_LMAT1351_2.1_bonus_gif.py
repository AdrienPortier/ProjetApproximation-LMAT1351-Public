import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D  # Importation de Line2D pour la légende

# Fonction pour calculer le point t sur la droite reliant P0 et P1
def point_on_line(t, P0, P1):
    return (1 - t) * P0 + t * P1


def B(t, P):
    if len(P) == 2:
        # Cas de la Bézier linéaire (2 points)
        return (1 - t) * P[0][0] + t * P[1][0], (1 - t) * P[0][1] + t * P[1][1]
    else:
        # Cas récursif pour la courbe Bézier de degré n
        return (1 - t) * np.array(B(t, P[:-1])) + t * np.array(B(t, P[1:]))
    
# Fonction pour calculer les points intermédiaires pour la récursion
def recursive_points(t, P):
    points = [P]  # Liste qui va contenir les sous-listes de points pour chaque niveau de récursion
    while len(P) > 1:
        P_new = []
        for i in range(len(P) - 1):
            P_new.append(point_on_line(t, P[i], P[i + 1]))
        points.append(P_new)  # Ajouter les nouveaux points à la liste
        P = P_new  # Passer au niveau suivant
    return points

# Liste des points de contrôle
liste_p = np.array([[-1,0], [-0.5,15], [0, -5], [2, 0], [1, 3],[4,5],[-2,8]])
liste_px = liste_p[:, 0]
liste_py = liste_p[:, 1]

t_vals = np.linspace(0, 1, 100)
frames_total = len(t_vals) + int(len(t_vals)*0.25)

colors = ["magenta", 'purple', 'orange', 'green', 'blue', 'darkred']

res_x, res_y = B(t_vals, liste_p)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(min(liste_px) - 1, max(liste_px) + 1)
ax.set_ylim(min(liste_py) - 1, max(liste_py) + 1)
ax.set_xlabel("x")
ax.set_ylabel("y")

for i in range(len(liste_p) - 1):
    ax.plot([liste_p[i][0], liste_p[i + 1][0]], [liste_p[i][1], liste_p[i + 1][1]], 'k--')

ax.scatter(liste_px[1:-1], liste_py[1:-1], color='black', s=75, marker='o', zorder=5)
ax.scatter(liste_px[0], liste_py[0], color='mediumseagreen', s=100, marker='*', label="Point de départ", zorder=5)
ax.scatter(liste_px[-1], liste_py[-1], color='royalblue', s=100, marker='*', label="Point d'arrivée", zorder=5)

# Mise à jour de l'image à chaque frame
def update(frame):
    if(frame >= len(t_vals)):
        t = 1
    else:
        t = t_vals[frame]

    points_at_t = recursive_points(t, liste_p)

    ax.clear()
    ax.set_xlim(min(liste_px) - 1, max(liste_px) + 1)
    ax.set_ylim(min(liste_py) - 1, max(liste_py) + 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


    ax.scatter(liste_px[1:-1], liste_py[1:-1], color='black', s=75, marker='o', zorder=5)
    ax.scatter(liste_px[0], liste_py[0], color='mediumseagreen', s=100, marker='*', label="Point de départ", zorder=5)
    ax.scatter(liste_px[-1], liste_py[-1], color='royalblue', s=100, marker='*', label="Point d'arrivée", zorder=5)

    #A chaque stade de récursion, relier les points calculés à chaque étape
    for level in range(len(points_at_t) - 1):
        points = points_at_t[level]
        for i in range(len(points) - 1):
            # Relier les points à la position t dans chaque niveau
            ax.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]], colors[level % len(colors)], lw=1.5)

    #Tracer la courbe Bézier finale
    ax.plot(res_x[:frame], res_y[:frame], color='red', lw=3)  # Épaisseur de ligne accrue pour plus de visibilité
    legend_proxy = Line2D([0], [0], color='red', lw=3)
    
    #Mettre à jour la légende avec la valeur actuelle de t
    ax.legend([legend_proxy, 
               ax.scatter(liste_px[0], liste_py[0], color='mediumseagreen', s=100, marker='*'),
               ax.scatter(liste_px[-1], liste_py[-1], color='royalblue', s=100, marker='*')],
              [f"Courbe de Bézier (t = {t:.2f})", "Point de départ", "Point d'arrivée"])

    return []

ani = FuncAnimation(fig, update, frames=frames_total, interval=20)

# Sauvegarde de l'animation en GIF
ani.save('animation_recursive_bezier.gif', writer=PillowWriter(fps=30))
