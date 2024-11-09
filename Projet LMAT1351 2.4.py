import numpy as np
import matplotlib.pyplot as plt

# Fonction pour calculer la courbe de Bézier quadratique
def bezier_quadratique(P0, P1, P2, t):
    x = (1 - t)**2 * P0[0] + 2 * (1 - t) * t * P1[0] + t**2 * P2[0]
    y = (1 - t)**2 * P0[1] + 2 * (1 - t) * t * P1[1] + t**2 * P2[1]
    return x, y

# Points de contrôle pour les 4 courbes de Bézier
P0_1 = [-1, 0]  # En haut à gauche
P1_1 = [-0.8, 1]
P2_1 = [0, -0.2]

P0_2 = [0, -0.2]  # En haut à droite
P1_2 = [0.8, 1]
P2_2 = [1, 0]

P0_3 = [1, 0]  # En bas à droite
P1_3 = [1.2, -0.8]
P2_3 = [0, -1.5]

P0_4 = [0, -1.5]  # En bas à gauche
P1_4 = [-1.2, -0.8]
P2_4 = [-1, 0]

# Paramètre t
t = np.linspace(0, 1, 1000)

# Calcul des points des courbes de Bézier
x1, y1 = bezier_quadratique(P0_1, P1_1, P2_1, t)
x2, y2 = bezier_quadratique(P0_2, P1_2, P2_2, t)
x3, y3 = bezier_quadratique(P0_3, P1_3, P2_3, t)
x4, y4 = bezier_quadratique(P0_4, P1_4, P2_4, t)

# Tracer le cœur
plt.figure(figsize=(8, 8))

# Tracer les 4 courbes de Bézier avec des couleurs différentes et des labels
plt.plot(x1, y1, 'r', label="Courbe 1")
plt.plot(x2, y2, 'g', label="Courbe 2")
plt.plot(x3, y3, 'b', label="Courbe 3")
plt.plot(x4, y4, 'm', label="Courbe 4")

# Ajouter les légendes
plt.legend()

# Ajuster l'aspect de l'image
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Cœur avec des courbes de Bézier quadratiques")
plt.savefig("2.4_heart.svg")
plt.show()
