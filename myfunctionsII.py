
import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from skimage.segmentation import active_contour
from skimage import img_as_float
from skimage.segmentation import active_contour, chan_vese
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.stats import entropy, skew
from skimage.filters import laplace
import tqdm
import plotly.express as px
import plotly.graph_objects as go
import ast


def step1(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_HSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    h_channel, s_channel, v_channel = cv2.split(img_HSV)

    img_gray = v_channel  # Usar el canal Value como imagen en gris

    v_channel_blur = cv2.medianBlur(img_gray, ksize=7)

    # Invertir si es necesario y verificar tamaño
    v_channel_blur, _ = invert_image_if_needed(v_channel_blur)

    mask = cv2.inRange(s_channel, 10, 255) & cv2.inRange(v_channel, 10, 255)
    h_channel_masked = cv2.bitwise_and(h_channel, h_channel, mask=mask)

    # Volver a unir los canales en HSV
    img_HSV2 = cv2.merge((h_channel_masked, s_channel, v_channel_blur))

    # Asegurar que todas las imágenes tengan el mismo tamaño
    target_size = img_rgb.shape[:2]  # Alto y ancho de la imagen RGB
    img_HSV2 = cv2.resize(img_HSV2, (target_size[1], target_size[0]))

    # Lista de imágenes
    img_list = [(img_rgb,'img_rgb'), (img_HSV,'img_HSV'), (img_gray,'img_gray'), (v_channel_blur,'v_channel_blur'), (img_HSV2,'img_HSV2',),(h_channel_masked,'h_channel_masked'),(s_channel,'s_channel')]
    
    return img_list, img_HSV2

def invert_image_if_needed(img_gray):
    # Calcular la media de los píxeles en las esquinas de la imagen para detectar el color del fondo
    corner_mean = np.mean([img_gray[0, 0], img_gray[0, -1], img_gray[-1, 0], img_gray[-1, -1]])

    # Si el fondo es claro (valor promedio alto), invertimos la imagen
    if corner_mean > 128:  # Umbral para decidir si el fondo es claro
        img_inverted = cv2.bitwise_not(img_gray)  # Invertir la imagen
        return img_inverted, True  # True indica que la imagen fue invertida
    else:
        return img_gray, False  # False indica que no se hizo inversión

def plotear_cosas(img_list, ncols,titulo):
    """
    Plotea las imágenes de una lista en un grid con n columnas.

    Parameters:
    - img_list: Lista de imágenes.
    - ncols: Número de columnas para el grid.
    """
    nrows = (len(img_list) + ncols - 1) // ncols  # Calcular filas necesarias
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5))
    plt.title(titulo)

    # Aplanar los ejes para fácil iteración
    axes = axes.flatten()

    for i, (imagen,nombre) in enumerate(img_list):
        cmap='gray'
        if nombre =='h_channel':
            cmap='hsv'
        axes[i].imshow(imagen, cmap=cmap)
        axes[i].set_title(f'Imagen {nombre}')  # Puedes personalizar el título
        axes[i].axis('off')

    # Ocultar subplots adicionales si sobran
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def calculate_indicators(hsv_img):
    from scipy.stats import entropy, skew
    """Calcula indicadores relevantes para los canales H, S, V."""
    # Separar los canales H, S, V
    h_channel, s_channel, v_channel = cv2.split(hsv_img)
    channels = [(h_channel, 'H'), (s_channel, 'S'), (v_channel, 'V')]

    # Lista para almacenar los resultados de cada canal
    indicators_list = []

    for channel, channel_name in channels:
        flat_channel = channel.flatten()

        # 1. Entropía
        hist, _ = np.histogram(flat_channel, bins=256, range=(0, 255))
        ent = entropy(hist + 1e-5)  # Evitar log(0)
        
        # 2. Desviación Estándar
        std = np.std(flat_channel)
        
        # 3. Rango Activo
        active_range = np.percentile(flat_channel, 95) - np.percentile(flat_channel, 5)
        
        # 4. Mediana y Asimetría
        med = np.median(flat_channel)
        skewness = skew(flat_channel)

        # 5. Contraste de Bordes (Laplaciano)
        edge_contrast = np.mean(np.abs(laplace(channel)))

        # Crear un diccionario con los resultados del canal
        info = {
            'Canal': channel_name,
            'Entropía': ent,
            'Desviación_Estándar': std,
            'Rango_Activo': active_range,
            'Mediana': med,
            'Asimetría': skewness,
            'Contraste_Bordes': edge_contrast
        }

        # Agregar el diccionario a la lista de resultados
        indicators_list.append(info)

    # Convertir la lista de resultados en un DataFrame
    indicators_df = pd.DataFrame(indicators_list)

    return indicators_df

def select_best_channel_for_minerals(indicators):
    """Selecciona el mejor canal para segmentar minerales basado en los indicadores."""
    # Pesos ajustados para segmentación de minerales
    weights = {
        'Entropía': 0.2, 
        'Rango_Activo': 0.2, 
        'Contraste_Bordes': 0.6, 
        'Asimetría': 0.1
    }

    # Calcular una puntuación ponderada para cada canal
    indicators['Puntuación'] = (
        indicators['Entropía'] * weights['Entropía'] +
        indicators['Rango_Activo'] * weights['Rango_Activo'] +
        indicators['Contraste_Bordes'] * weights['Contraste_Bordes'] +
        abs(indicators['Asimetría']) * weights['Asimetría']
    )

    # Seleccionar el canal con la puntuación más alta
    best_channel = indicators.loc[indicators['Puntuación'].idxmax(), 'Canal']
    return best_channel

# UMBRALIZACION

def otsu_thresholding(hsv_img):
    """
    Aplica umbralización Otsu al canal seleccionado.
    """
    h_channel, s_channel, v_channel = cv2.split(hsv_img)
    channels = [(h_channel, 'H'), (s_channel, 'S'), (v_channel, 'V')]
    lista_otsu=[]
    for (channel,nombre) in channels:

    # Aplicar umbralización Otsu
        _, otsu_result = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lista_otsu.append((otsu_result,nombre))
    return lista_otsu

# CONTORNOS

def dibujar_contornos(lista_otsu, titulo_imagen):  
    """
    Dibuja y muestra los contornos sobre las imágenes umbralizadas.
    
    Parameters:
    - lista_otsu: Lista de tuplas (imagen, nombre), donde la imagen está en escala de grises.

    Returns:
    - contornos_dict: Diccionario con los nombres de las imágenes como claves y sus contornos como valores.
    """
    n = len(lista_otsu)  # Número de imágenes umbralizadas
    if n == 0:
        print("Lista vacía. No hay imágenes para procesar.")
        return {}

    fig, axes = plt.subplots(1, n, figsize=(15, 10))
    plt.title(titulo_imagen)
    if n == 1:
        axes = [axes]  # Asegurar que 'axes' sea iterable en caso de una sola imagen

    contornos_dict = {}

    for i, (img, name) in enumerate(lista_otsu):
        # Encontrar contornos
        contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Guardar contornos en el diccionario
        contornos_dict[name] = contornos

        # Convertir la imagen umbralizada a RGB para dibujar los contornos en color
        img_with_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(img_with_contours, contornos, -1, (255, 0, 0), 2)  # Contornos en rojo

        # Mostrar la imagen con contornos
        axes[i].imshow(img_with_contours)
        axes[i].set_title(f'Contornos {name}')
        axes[i].axis('off')

    '''    plt.tight_layout()
        plt.show()'''

    return contornos_dict


def extract_contour_info(path, best_channel, contours_dict, indicators):
    """
    Extrae información de los contornos del canal seleccionado y la añade al diccionario de entrenamiento.

    Parameters:
    - path: Ruta de la imagen procesada.
    - best_channel: Canal seleccionado manualmente ('H', 'S', 'V').
    - contours_dict: Diccionario con los contornos por canal.
    - indicators: DataFrame con los indicadores por canal.

    Returns:
    - contour_info_list: Lista con la información de cada contorno.
    """
    # Obtener los contornos del canal seleccionado
    contours = contours_dict[best_channel]

    # Lista para acumular la información de los contornos
    contour_info_list = []

    # Iterar sobre cada contorno
    for i, contour in enumerate(contours):
        # Convertir el contorno a lista limpia de coordenadas (sin saltos de línea ni problemas)
        contour_list = contour.reshape(-1, 2).tolist()

        # Calcular las características del contorno
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        rect_area = w * h
        extent = area / rect_area if rect_area != 0 else 0
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

        # Calcular el centroide
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Guardar la información del contorno
        contour_info = {
            'Imagen': path,
            'Canal': best_channel,
            'Canal_info': indicators[indicators['Canal'] == best_channel].to_dict('records'),
            'Contorno_ID': i,
            'Contorno': contour,  # Almacenar como lista limpia de coordenadas
            'Area': area,
            'Perimetro': perimeter,
            'Aspect_Ratio': aspect_ratio,
            'Extent': extent,
            'Compactness': compactness,
            'Centroid': (cx, cy),
            'BoundingBox': (x, y, w, h)
        }

        # Agregar la información del contorno a la lista
        contour_info_list.append(contour_info)

    return contour_info_list


def plot_contours_and_metrics(img_rgb, contours_dict):
    """
    Visualiza los contornos de diferentes canales sobre la imagen original en RGB y 
    muestra las métricas asociadas para la selección manual del mejor canal.

    Parameters:
    - img_rgb: Imagen original en RGB.
    - contours_dict: Diccionario con contornos por canal (clave: canal, valor: contornos).
    """
    n_channels = len(contours_dict)
    fig, axes = plt.subplots(1, n_channels, figsize=(18, 12))  # Ajuste dinámico según el número de canales

    # Dibujar contornos sobre la imagen original en RGB
    for i, (channel_name, contours) in enumerate(contours_dict.items()):
        img_with_contours = img_rgb.copy()
        cv2.drawContours(img_with_contours, contours, -1, (255, 0, 0), 2)  # Contornos en rojo
        
        # Mostrar la imagen con los contornos superpuestos
        axes[i].imshow(img_with_contours)
        axes[i].set_title(f'Contornos - {channel_name}')
        axes[i].axis('off')


    plt.tight_layout()
    plt.show()

def manual_selection():
    """
    Permite la selección manual del mejor canal.
    """
    selected_channel = input("Escribe el nombre del canal seleccionado como mejor opción (S, H o V): ")
    print(f"Canal seleccionado: {selected_channel}")
    return selected_channel

def visualize_all_contours(df):
    """
    Visualiza todos los contornos y bounding boxes asociados a cada imagen en la misma foto.
    
    Parameters:
    - df: DataFrame con información de los contornos, canales y bounding boxes.
    """
    # Agrupar las filas del DataFrame por cada imagen
    grouped = df.groupby('Imagen')

    for img_path, group in grouped:
        # Cargar la imagen en RGB
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Dibujar todos los contornos y bounding boxes de esta imagen
        for idx, row in group.iterrows():
            # Extraer contorno y bounding box
            contour = np.array(row['Contorno'])
            x, y, w, h = row['BoundingBox']

            # Dibujar el contorno en rojo
            cv2.drawContours(image_rgb, [contour], -1, (255, 0, 0), 2)

            # Dibujar la bounding box en verde
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Dibujar el centroide en azul
            cx, cy = row['Centroid']
            cv2.circle(image_rgb, (cx, cy), 5, (0, 0, 255), -1)

            # Añadir el área como texto en la imagen
            area = row['Area']
            cv2.putText(
                image_rgb,
                f'Area: {int(area)}',
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Mostrar la imagen con todos los contornos y bounding boxes superpuestos
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(f"Contornos para Imagen: {img_path}")
        plt.show()

        
# OPERACIONES MORFOLOGICAS

def morphological_opening(umbralized_channels, kernel_size=(5, 5)):
    """Aplica apertura morfológica (erosión seguida de dilatación)."""
    lista_opened = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    for (channel, name) in umbralized_channels:
        opened = cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel)
        lista_opened.append((opened, name))
    return lista_opened

def morphological_closing(umbralized_channels, kernel_size=(5, 5)):
    """Aplica cierre morfológico (dilatación seguida de erosión)."""
    lista_closed = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    for (channel, name) in umbralized_channels:
        closed = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel)
        lista_closed.append((closed, name))
    return lista_closed

def morphological_erosion(umbralized_channels, kernel_size=(5, 5)):
    """Aplica erosión morfológica."""
    lista_eroded = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    for (channel, name) in umbralized_channels:
        eroded = cv2.erode(channel, kernel, iterations=1)
        lista_eroded.append((eroded, name))
    return lista_eroded

def morphological_dilation(umbralized_channels, kernel_size=(5, 5)):
    """Aplica dilatación morfológica."""
    lista_dilated = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    for (channel, name) in umbralized_channels:
        dilated = cv2.dilate(channel, kernel, iterations=1)
        lista_dilated.append((dilated, name))
    return lista_dilated

def morphological_gradient(umbralized_channels, kernel_size=(5, 5)):
    """Aplica gradiente morfológico (dilatación - erosión)."""
    lista_gradient = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    for (channel, name) in umbralized_channels:
        gradient = cv2.morphologyEx(channel, cv2.MORPH_GRADIENT, kernel)
        lista_gradient.append((gradient, name))
    return lista_gradient


def generate_masks_from_dataframe(df, save_dir="masks/"):
    """
    Genera una máscara binaria para cada imagen en el DataFrame con base en sus contornos.
    
    Parameters:
    - df: DataFrame que contiene los contornos y rutas de las imágenes.
    - save_dir: Directorio donde se guardarán las máscaras.
    """
    os.makedirs(save_dir, exist_ok=True)  # Crear el directorio si no existe

    grouped = df.groupby('Imagen')  # Agrupar por nombre de la imagen

    for img_path, group in tqdm(grouped):
        # Leer la imagen para obtener sus dimensiones
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error al leer la imagen: {img_path}")
            continue
        height, width = image.shape[:2]

        # Crear una máscara vacía
        mask = np.zeros((height, width), dtype=np.uint8)

        # Dibujar cada contorno en la máscara
        for _, row in group.iterrows():
            # Convertir el contorno a formato numpy adecuado
            contour = np.array(eval(row['Contorno']), dtype=np.int32)
            contour = contour.reshape((-1, 1, 2))  # Asegurar la forma adecuada

            # Dibujar el contorno en la máscara
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Guardar la máscara
        mask_name = os.path.join(save_dir, os.path.basename(img_path).replace('.jpg', '_mask.png'))
        cv2.imwrite(mask_name, mask)
        print(f"Máscara guardada en: {mask_name}")


def expand_channel_info(df):
    """
    Expande la columna 'Canal_info' para agregar sus contenidos como nuevas columnas.
    """
    # Lista para almacenar los diccionarios convertidos a filas
    new_columns = []

    for i, row in df.iterrows():
        # Leer el contenido de la columna 'Canal_info' y convertirlo de string a lista de dict
        try:
            canal_info_list = ast.literal_eval(str(row['Canal_info']))  # Asegura que sea una lista de dict
            if isinstance(canal_info_list, list) and len(canal_info_list) > 0:
                canal_info_dict = canal_info_list[0]  # Extraer el primer diccionario
            else:
                canal_info_dict = {}
        except (ValueError, SyntaxError):
            canal_info_dict = {}

        # Añadir el diccionario convertido a la lista
        new_columns.append(canal_info_dict)

    # Crear un DataFrame con las nuevas columnas
    new_columns_df = pd.DataFrame(new_columns)

    # Concatenar con el DataFrame original
    df_expanded = pd.concat([df.reset_index(drop=True), new_columns_df], axis=1)

    # Eliminar la columna 'Canal_info' original si ya no es necesaria
    df_expanded = df_expanded.drop(columns=['Canal_info'], errors='ignore')

    return df_expanded


def extract_contour_info_for_all_channels(path, contours_dict, indicators):
    """
    Extrae información de los contornos para los 3 canales (H, S, V) y construye la data para el modelo.

    Parameters:
    - path: Ruta de la imagen procesada.
    - contours_dict: Diccionario con los contornos por canal.
    - indicators: DataFrame con los indicadores por canal.

    Returns:
    - contour_info_list: Lista con la información de todos los contornos de los 3 canales.
    """
    # Lista para acumular la información de todos los contornos
    contour_info_list = []

    # Iterar sobre cada canal (H, S, V)
    for channel_name, contours in contours_dict.items():
        # Iterar sobre cada contorno en el canal actual
        for i, contour in enumerate(contours):
            # Convertir el contorno a una lista limpia de coordenadas
            contour_list = contour.reshape(-1, 2).tolist()

            # Calcular las características del contorno
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            rect_area = w * h
            extent = area / rect_area if rect_area != 0 else 0
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

            # Calcular el centroide
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Obtener la información del canal desde el DataFrame de indicadores
            canal_info = indicators[indicators['Canal'] == channel_name].to_dict('records')

            # Guardar la información del contorno
            contour_info = {
                'Imagen': path,
                #'Canal': channel_name,
                'Canal_info': canal_info,
                'Contorno_ID': i,
                'Contorno': contour_list,  # Guardar como lista limpia
                'Area': area,
                'Perimetro': perimeter,
                'Aspect_Ratio': aspect_ratio,
                'Extent': extent,
                'Compactness': compactness,
                'Centroid': (cx, cy),
                'BoundingBox': (x, y, w, h)
            }

            # Agregar la información del contorno a la lista
            contour_info_list.append(contour_info)

    return contour_info_list

def procesar_imagen_app(image_path,modelo,encoder):
    """
    Aplica preprocesamiento a la imagen y selecciona el mejor canal.
    """
    lista, img_HSV2 = step1(image_path)
    indicators = calculate_indicators(img_HSV2)  

    # Umbralización y operaciones morfológicas
    lista_otsu = otsu_thresholding(img_HSV2)
    lista_dilation = morphological_dilation(lista_otsu, kernel_size=(5, 5))
    lista_opened = morphological_opening(lista_dilation, kernel_size=(7, 7))
    lista_closing = morphological_closing(lista_opened, kernel_size=(5, 5))
    lista_erosion = morphological_erosion(lista_closing, kernel_size=(1, 1))

    # Dibujar contornos en cada operación morfológica
    
    contornos_dilation = dibujar_contornos(lista_dilation, 'Dilatación')
    contornos_opened = dibujar_contornos(lista_opened, 'Dilatación + Apertura')
    contornos_closing = dibujar_contornos(lista_closing, 'Dilatación + Apertura + Cierre')
    contornos_erosion = dibujar_contornos(lista_erosion, 'Dilatación + Apertura + Cierre + Erosión')

    img_rgb = lista[0][0]

    # Inicializa un DataFrame vacío para acumular los resultados
    contour_df = pd.DataFrame()

    contour_info_list = extract_contour_info_for_all_channels(image_path, contornos_erosion, indicators)
    

    # Convertir la lista de diccionarios en un DataFrame temporal
    temp_df = pd.DataFrame(contour_info_list)
    

    # Concatenar el DataFrame temporal con el acumulado
    contours_app = pd.concat([contour_df, temp_df], axis=0, ignore_index=True)

    # Guardar el DataFrame completo en un archivo CSV
    contours_app.to_csv('contour_data.csv', index=False)
    
    df_expanded= expand_channel_info(contours_app)

    print(df_expanded.head(1))
    print(df_expanded.columns)

    features = ['Area', 'Perimetro',
       'Aspect_Ratio', 'Extent', 'Compactness',
       'Entropía', 'Desviación_Estándar', 'Rango_Activo', 'Mediana',
       'Asimetría', 'Contraste_Bordes']

# Variables X (características) y y (Canal a predecir)
    X = df_expanded[features]

    canal_pred = modelo.predict(X)[0]
    canal_decodificado = encoder.inverse_transform([canal_pred])[0]

    return img_rgb, img_HSV2, canal_decodificado, df_expanded,lista_erosion

def dibujar_contornos_en_imagen(img_rgb, df_expanded, canal_predicho):
    """
    Dibuja los contornos del canal predicho sobre la imagen RGB.
    
    Parameters:
    - img_rgb: La imagen original en RGB.
    - df_expanded: DataFrame con la información de los contornos.
    - canal_predicho: Canal seleccionado por el modelo ('H', 'S', 'V').

    Returns:
    - img_with_contours: Imagen RGB con los contornos dibujados.
    """
    # Filtrar los contornos correspondientes al canal predicho
    df_canal = df_expanded[df_expanded['Canal'] == canal_predicho]

    # Crear una copia de la imagen RGB para dibujar los contornos
    img_with_contours = img_rgb.copy()



    # Dibujar cada contorno en la imagen
    for _, row in df_canal.iterrows():
        # Obtener el contorno desde el DataFrame
        contour = np.array(row['Contorno'], dtype=np.int32)

        # Dibujar el contorno en rojo sobre la imagen
        cv2.drawContours(img_with_contours, [contour], -1, (255, 0, 0), 2)

    return img_with_contours


def generate_interactive_plot(img_rgb, df_expanded, canal):
    """
    Genera una gráfica interactiva con Plotly para un canal específico.
    """
    fig = px.imshow(img_rgb)
    fig.update_layout(coloraxis_showscale=False)  # Sin barra de color

    # Filtrar los contornos y bounding boxes del canal
    canal_contours = df_expanded[df_expanded['Canal'] == canal]

    # Dibujar cada contorno
    for _, row in canal_contours.iterrows():
        contour = np.array(row['Contorno'])

        # Dibujar el contorno en rojo
        fig.add_trace(
            go.Scatter(
                x=contour[:, 0],
                y=contour[:, 1],
                mode='lines',
                name=f"Contorno {row['Contorno_ID']}",
                hovertemplate=(
                    f"ID: {row['Contorno_ID']}<br>"
                    f"Área: {row['Area']}<br>"
                    f"Perímetro: {row['Perimetro']}<br>"
                    f"Aspect Ratio: {round(row['Aspect_Ratio'], 2)}<br>"
                    f"Extent: {round(row['Extent'], 2)}<br>"
                    f"Compactness: {round(row['Compactness'], 2)}"
                ),
                line=dict(color='red'),
                showlegend=False
            )
        )

        # Dibujar bounding box en verde claro
        x, y, w, h = row['BoundingBox']
        fig.add_shape(
            type="rect",
            x0=x, y0=y, x1=x + w, y1=y + h,
            line=dict(color="lightgreen", width=2)
        )

    fig.update_layout(
        title=f"Visualización Interactiva - Canal {canal}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    return fig














