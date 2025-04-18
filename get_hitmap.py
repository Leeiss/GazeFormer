
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize

image_path = '000000491881.jpg'
img = Image.open(image_path)
img = np.array(img)
height, width, _ = img.shape

original_width = 512
original_height = 320

scale_x = width / original_width
scale_y = height / original_height

fixations_data = [
    {"X": [256.0, 409.7572021484375, 412.41876220703125],
     "Y": [160.0, 194.03179931640625, 185.96875],
     "T": [224.1181640625, 221.8061981201172, 253.4873809814453]},

]

heatmap_simple = np.zeros((height, width))
heatmap_weighted = np.zeros((height, width))

for fixation in fixations_data:
    for x, y, t in zip(fixation["X"], fixation["Y"], fixation["T"]):
        x_scaled = int(round(x * scale_x))
        y_scaled = int(round(y * scale_y))
        if 0 <= x_scaled < width and 0 <= y_scaled < height:
            heatmap_simple[y_scaled, x_scaled] += 1
            heatmap_weighted[y_scaled, x_scaled] += t / 100.0

heatmap_simple = gaussian_filter(heatmap_simple, sigma=10)
heatmap_weighted = gaussian_filter(heatmap_weighted, sigma=10)

def normalize(hmap):
    hmap = np.clip(hmap, 0, np.max(hmap))
    hmap = (hmap / np.max(hmap)) * 255
    return np.uint8(hmap)

heatmap_simple = normalize(heatmap_simple)
heatmap_weighted = normalize(heatmap_weighted)

def save_heatmap(base_img, heatmap, filename):
    plt.figure(figsize=(10, 6))
    plt.imshow(base_img)
    plt.imshow(heatmap, cmap='jet', alpha=0.6)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

cmap = plt.cm.get_cmap('Blues')

def normalize_time(T):
    norm = Normalize(vmin=min(T), vmax=max(T))
    return [cmap(norm(t)) for t in T]

def save_gaze_trajectory_with_time_gradient(base_img, fixation_list, filename):
    plt.figure(figsize=(10, 6))
    plt.imshow(base_img)

    for fixation in fixation_list:
        X = [x * scale_x for x in fixation['X']]
        Y = [y * scale_y for y in fixation['Y']]
        T = fixation['T']
        colors = normalize_time(T)

        for i in range(1, len(X)):
            plt.plot([X[i-1], X[i]], [Y[i-1], Y[i]], color=colors[i-1], lw=2)

        for i, (x, y) in enumerate(zip(X, Y)):
            plt.scatter(x, y, color=colors[i], s=100, zorder=10)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# Сохранение результатов
save_heatmap(img, heatmap_simple, 'heatmap_simple1.png')
save_heatmap(img, heatmap_weighted, 'heatmap_time_weighted1.png')
save_gaze_trajectory_with_time_gradient(img, fixations_data, 'gaze_trajectory1.png')
