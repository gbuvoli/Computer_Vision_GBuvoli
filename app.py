from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import joblib
import plotly.graph_objects as go
import os
from myfunctionsII import *

# Inicializa la aplicación Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./static/uploads"

# Cargar el modelo y el LabelEncoder
modelo = joblib.load("modelo_mejor_canal.pkl")
encoder = joblib.load("label_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Guardar la imagen subida
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Procesar la imagen y obtener resultados
            img_rgb, img_HSV, canal_predicho, df_expanded, lista_erosion = procesar_imagen_app(file_path, modelo, encoder)

            # Generar las gráficas para los tres canales
            fig_H = generate_interactive_plot(img_rgb, df_expanded, 'H')
            fig_S = generate_interactive_plot(img_rgb, df_expanded, 'S')
            fig_V = generate_interactive_plot(img_rgb, df_expanded, 'V')

            # Convertir las gráficas a HTML
            graph_html_H = fig_H.to_html(full_html=False)
            graph_html_S = fig_S.to_html(full_html=False)
            graph_html_V = fig_V.to_html(full_html=False)

            # Renderizar la plantilla HTML con las gráficas
            return render_template(
                "index.html",
                canal=canal_predicho,
                original=file.filename,
                graph_html_H=graph_html_H,
                graph_html_S=graph_html_S,
                graph_html_V=graph_html_V
            )

    return render_template("index.html")





if __name__ == "__main__":
    if not os.path.exists("./static/uploads"):
        os.makedirs("./static/uploads")
    app.run(debug=True)
