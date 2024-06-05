import numpy as np
import cv2

def pca_decomposition(image, n_components):
    """
    Выполняет декомпозицию изображения с помощью метода главных компонент (PCA).
    
    Параметры:
    - image: исходное изображение (numpy array)
    - n_components: количество главных компонент
    
    Возвращает:
    - reconstructed_image: реконструированное изображение после декомпозиции
    """
    # Преобразование изображения в одномерный массив
    flat_image = image.reshape(-1, image.shape[-1])
    
    # Вычисление матрицы ковариации
    cov_matrix = np.cov(flat_image.T)
    
    # Вычисление собственных векторов и собственных значений матрицы ковариации
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Сортировка собственных векторов в порядке убывания соответствующих собственных значений
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Выбор первых n_components собственных векторов
    principal_components = eigenvectors[:, :n_components]
    
    # Проецирование данных на новое пространство главных компонент
    projected_data = np.dot(flat_image, principal_components)
    
    # Реконструкция изображения из главных компонент
    reconstructed_data = np.dot(projected_data, principal_components.T)
    reconstructed_image = reconstructed_data.reshape(image.shape)
    
    return reconstructed_image

def svd_decomposition(image, n_components):
    """
    Выполняет декомпозицию изображения с помощью сингулярного разложения (SVD).
    
    Параметры:
    - image: исходное изображение (numpy array)
    - n_components: количество компонент
    
    Возвращает:
    - reconstructed_image: реконструированное изображение после декомпозиции
    """
    # Преобразование изображения в матрицу
    matrix = image.reshape(image.shape[0], -1)
    
    # Выполнение SVD разложения
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    
    # Выбор первых n_components компонент
    U = U[:, :n_components]
    S = S[:n_components]
    VT = VT[:n_components, :]
    
    # Реконструкция изображения из выбранных компонент
    reconstructed_matrix = np.dot(U, np.dot(np.diag(S), VT))
    reconstructed_image = reconstructed_matrix.reshape(image.shape)
    
    return reconstructed_image

def ica_decomposition(image, n_components):
    """
    Выполняет декомпозицию изображения с помощью независимых компонент (ICA).
    
    Параметры:
    - image: исходное изображение (numpy array)
    - n_components: количество независимых компонент
    
    Возвращает:
    - reconstructed_image: реконструированное изображение после декомпозиции
    """
    # Преобразование изображения в одномерный массив
    flat_image = image.reshape(-1, image.shape[-1])
    
    # Центрирование данных
    mean = np.mean(flat_image, axis=0)
    centered_data = flat_image - mean
    
    # Вычисление матрицы ковариации
    cov_matrix = np.cov(centered_data.T)
    
    # Вычисление собственных векторов и собственных значений матрицы ковариации
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Вычисление матрицы отбеливания
    whitening_matrix = np.dot(eigenvectors, np.diag(eigenvalues**(-0.5)))
    
    # Отбеливание данных
    whitened_data = np.dot(centered_data, whitening_matrix)
    
    # Инициализация матрицы смешивания
    mixing_matrix = np.random.randn(n_components, whitened_data.shape[1])
    
    # Итеративное обучение ICA
    max_iterations = 100
    for _ in range(max_iterations):
        # Вычисление независимых компонент
        independent_components = np.dot(whitened_data, mixing_matrix.T)
        
        # Вычисление функции активации (негауссовости)
        g = np.tanh(independent_components)
        
        # Вычисление производной функции активации
        g_derivative = 1 - g**2
        
        # Обновление матрицы смешивания
        mixing_matrix = mixing_matrix + np.dot(g, whitened_data) / whitened_data.shape[0] - np.dot(g_derivative.mean(axis=0)[:, np.newaxis], mixing_matrix)
    
    # Реконструкция изображения из независимых компонент
    reconstructed_data = np.dot(independent_components, mixing_matrix) + mean
    reconstructed_image = reconstructed_data.reshape(image.shape)
    
    return reconstructed_image