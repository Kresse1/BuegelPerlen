import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import argparse
import os

def apply_edge_enhancement(image):
    edges = image.filter(ImageFilter.FIND_EDGES)
    enhanced = Image.blend(image, edges, 0.3)
    return enhanced

def quantize_colors(image, n_colors):
    pixels = np.array(image).reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    quantized = kmeans.cluster_centers_.astype(np.uint8)[labels]
    return Image.fromarray(quantized.reshape(*image.size[::-1], 3))

def process_image(input_path, output_path, n_colors=48):
    colors = [
        "#fda0e1", "#ea59bc", "#fb5176", "#e90e10", "#da0100", "#b50305", "#ffc0e6",
        "#cab0f8", "#c31fa5", "#af4be5", "#6513ae", "#180274", "#fedba3", "#f0733e",
        "#842f04", "#861f0d", "#6d3013", "#4d0100", "#ffffff", "#768481", "#6a7070",
        "#222325", "#000000", "#d97bad", "#f7cfb2", "#ffc4af", "#fec825", "#fdac38",
        "#df7029", "#f3f490", "#efff59", "#e8f62d", "#fcda3e", "#e4c21a", "#f7dc14",
        "#f3ca11", "#03aaae", "#94efb8", "#b0f14c", "#12e66d", "#29b679", "#0e8774",
        "#89b8d6", "#34afeb", "#0ca6c9", "#0ea4e1", "#177ed5", "#2554ad"
    ]

    # Öffne das Bild und konvertiere Transparenz zu Weiß
    img = Image.open(input_path).convert('RGBA')
    background = Image.new('RGBA', img.size, (255, 255, 255))
    img = Image.alpha_composite(background, img).convert('RGB')

    img_resized = img.resize((200, 200), Image.LANCZOS)
    img_enhanced = apply_edge_enhancement(img_resized)
    img_quantized = quantize_colors(img_enhanced, n_colors)
    img_final = img_quantized.resize((50, 50), Image.NEAREST)
    img_array = np.array(img_final)

    custom_cmap = ListedColormap(colors)

    plt.figure(figsize=(20, 20))
    plt.imshow(img_array, cmap=custom_cmap, interpolation='nearest')
    
    # Raster hinzufügen
    for x in range(50):
        plt.axvline(x - 0.5, color='black', linewidth=0.5, alpha=0.5)
    for y in range(50):
        plt.axhline(y - 0.5, color='black', linewidth=0.5, alpha=0.5)
    
    # Koordinaten hinzufügen
    for y in range(50):
        for x in range(50):
            plt.text(x, y, f'{x},{y}', ha='center', va='center', fontsize=4, color='black')
    
    plt.xlim(-0.5, 49.5)
    plt.ylim(49.5, -0.5)
    plt.axis('off')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    print(f"Verarbeitetes Bild mit Raster und Koordinaten wurde als {output_path} gespeichert.")

def main():
    parser = argparse.ArgumentParser(description='Verarbeite ein Bild und füge Raster und Koordinaten hinzu.')
    parser.add_argument('input_path', type=str, help='Pfad zum Eingabebild')
    parser.add_argument('-o', '--output_path', type=str, help='Pfad zum Ausgabebild (optional)')
    args = parser.parse_args()

    input_path = args.input_path
    if args.output_path:
        output_path = args.output_path
    else:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}_processed.png"

    process_image(input_path, output_path)

if __name__ == "__main__":
    main()