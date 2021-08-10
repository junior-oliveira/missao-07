import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils import db
from sklearn.decomposition import PCA

# Realiza a redução de dimensionalidade com UMAP
def get_dados_reducao(dados, n_components = 5):
    import umap
    dados = dados
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reducer.fit(dados)
    dados_transformados = reducer.transform(dados)
    return dados_transformados

def reducao_pca(X):
    pca = PCA(n_components=20)
    X_transform = pca.fit_transform(X)
    return X_transform