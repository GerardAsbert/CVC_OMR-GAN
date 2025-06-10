import os
from PIL import Image

directories = ['../Datasets/Printed/deepscores_symbols_reduced'] #'./dataset1', './dataset2',

TARGET_CLASSES = {'accidentalflat', 'accidentalsharp', 'cleff', 'clefg'}

# Función para listar todas las imágenes en los directorios especificados y obtener las clases
def list_images_in_directories(directories, target_classes):
    all_images = {}
    seen_images = set()
    total_images = 0
    for data_dir in directories:
        if not os.path.exists(data_dir):
            print(f"Directorio no existe: {data_dir}")
            continue
        print(f"Procesando directorio: {data_dir}")
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            class_lower = class_name.lower()
            if class_lower in target_classes and os.path.isdir(class_dir):
                if class_lower not in all_images:
                    all_images[class_lower] = []
                png_count = 0
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith('.png'):
                        img_path = os.path.join(class_dir, img_file)
                        if img_path not in seen_images:
                            all_images[class_lower].append(img_path)
                            seen_images.add(img_path)
                            png_count += 1
                            total_images += 1
                print(f"Archivos .png encontrados en {class_dir}: {png_count}")
            else:
                print(f"Clase no encontrada o no es un directorio: {class_dir}")
    return all_images, total_images

# Ejecutar la función y obtener todas las imágenes y clases
all_images, total_images = list_images_in_directories(directories, TARGET_CLASSES)

# Mostrar el total de imágenes encontradas por clase
for class_name, images in all_images.items():
    print(f"Clase: {class_name}, Total de imágenes: {len(images)}")

# Mostrar el total de imágenes encontradas
print(f"Total de imágenes encontradas: {total_images}")

# Opcional: Guardar la lista de imágenes en un archivo de texto
output_file = 'all_images.txt'
with open(output_file, 'w') as f:
    for class_name, images in all_images.items():
        for img_path in images:
            f.write(f"{class_name}\t{img_path}\n")
    print(f"Lista de imágenes guardada en {output_file}")
