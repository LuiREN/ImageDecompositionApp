from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
import base64
from decomposition import pca_decomposition, svd_decomposition, ica_decomposition

app = Flask(__name__)

# Конфигурация папки для загрузки изображений
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'Изображени не загружено'
    
    image_file = request.files['image']
    algorithm = request.form['algorithm']
    
    # Сохранение загруженного изображения
    filename = image_file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)
    
    # Загрузка изображения с помощью OpenCV
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Выполнение декомпозиции изображения
    if algorithm == 'pca':
        decomposed_image = pca_decomposition(image, n_components=50)
    elif algorithm == 'svd':
        decomposed_image = svd_decomposition(image, n_components=50)
    elif algorithm == 'ica':
        decomposed_image = ica_decomposition(image, n_components=50)
    
    # Сохранение исходного изображения в буфер
    original_buffer = BytesIO()
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(original_buffer, format='png')
    original_buffer.seek(0)
    original_image_url = base64.b64encode(original_buffer.getvalue()).decode('utf-8')
    
    # Сохранение декомпозированного изображения в буфер
    decomposed_buffer = BytesIO()
    plt.imshow(decomposed_image)
    plt.axis('off')
    plt.savefig(decomposed_buffer, format='png')
    decomposed_buffer.seek(0)
    decomposed_image_url = base64.b64encode(decomposed_buffer.getvalue()).decode('utf-8')
    
    # Отображение результатов
    return render_template('result.html',
                           original_image=f'data:image/png;base64,{original_image_url}',
                           decomposed_image=f'data:image/png;base64,{decomposed_image_url}',
                           algorithm=algorithm.upper())

if __name__ == '__main__':
    app.run(debug=True)