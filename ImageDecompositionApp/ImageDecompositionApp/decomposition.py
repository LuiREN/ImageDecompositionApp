import numpy as np
import cv2

def pca_decomposition(image, n_components):
    """
    ��������� ������������ ����������� � ������� ������ ������� ��������� (PCA).
    
    ���������:
    - image: �������� ����������� (numpy array)
    - n_components: ���������� ������� ���������
    
    ����������:
    - reconstructed_image: ������������������ ����������� ����� ������������
    """
    # �������������� ����������� � ���������� ������
    flat_image = image.reshape(-1, image.shape[-1])
    
    # ���������� ������� ����������
    cov_matrix = np.cov(flat_image.T)
    
    # ���������� ����������� �������� � ����������� �������� ������� ����������
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # ���������� ����������� �������� � ������� �������� ��������������� ����������� ��������
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # ����� ������ n_components ����������� ��������
    principal_components = eigenvectors[:, :n_components]
    
    # ������������� ������ �� ����� ������������ ������� ���������
    projected_data = np.dot(flat_image, principal_components)
    
    # ������������� ����������� �� ������� ���������
    reconstructed_data = np.dot(projected_data, principal_components.T)
    reconstructed_image = reconstructed_data.reshape(image.shape)
    
    return reconstructed_image

def svd_decomposition(image, n_components):
    """
    ��������� ������������ ����������� � ������� ������������ ���������� (SVD).
    
    ���������:
    - image: �������� ����������� (numpy array)
    - n_components: ���������� ���������
    
    ����������:
    - reconstructed_image: ������������������ ����������� ����� ������������
    """
    # �������������� ����������� � �������
    matrix = image.reshape(image.shape[0], -1)
    
    # ���������� SVD ����������
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    
    # ����� ������ n_components ���������
    U = U[:, :n_components]
    S = S[:n_components]
    VT = VT[:n_components, :]
    
    # ������������� ����������� �� ��������� ���������
    reconstructed_matrix = np.dot(U, np.dot(np.diag(S), VT))
    reconstructed_image = reconstructed_matrix.reshape(image.shape)
    
    return reconstructed_image

def ica_decomposition(image, n_components):
    """
    ��������� ������������ ����������� � ������� ����������� ��������� (ICA).
    
    ���������:
    - image: �������� ����������� (numpy array)
    - n_components: ���������� ����������� ���������
    
    ����������:
    - reconstructed_image: ������������������ ����������� ����� ������������
    """
    # �������������� ����������� � ���������� ������
    flat_image = image.reshape(-1, image.shape[-1])
    
    # ������������� ������
    mean = np.mean(flat_image, axis=0)
    centered_data = flat_image - mean
    
    # ���������� ������� ����������
    cov_matrix = np.cov(centered_data.T)
    
    # ���������� ����������� �������� � ����������� �������� ������� ����������
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # ���������� ������� �����������
    whitening_matrix = np.dot(eigenvectors, np.diag(eigenvalues**(-0.5)))
    
    # ����������� ������
    whitened_data = np.dot(centered_data, whitening_matrix)
    
    # ������������� ������� ����������
    mixing_matrix = np.random.randn(n_components, whitened_data.shape[1])
    
    # ����������� �������� ICA
    max_iterations = 100
    for _ in range(max_iterations):
        # ���������� ����������� ���������
        independent_components = np.dot(whitened_data, mixing_matrix.T)
        
        # ���������� ������� ��������� (�������������)
        g = np.tanh(independent_components)
        
        # ���������� ����������� ������� ���������
        g_derivative = 1 - g**2
        
        # ���������� ������� ����������
        mixing_matrix = mixing_matrix + np.dot(g, whitened_data) / whitened_data.shape[0] - np.dot(g_derivative.mean(axis=0)[:, np.newaxis], mixing_matrix)
    
    # ������������� ����������� �� ����������� ���������
    reconstructed_data = np.dot(independent_components, mixing_matrix) + mean
    reconstructed_image = reconstructed_data.reshape(image.shape)
    
    return reconstructed_image