import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def ler_base_treinamento():
  dados_treinamento = pd.read_csv('bases_de_dados/train.csv')
  dados_treinamento = dados_treinamento.drop('id', axis=1)
  return dados_treinamento

def ler_base_teste():
  dados_teste = pd.read_csv('bases_de_dados/test.csv')
  return dados_teste

def ler_sample_submission():
  dados_trainamento = pd.read_csv('bases_de_dados/sample_submission.csv')
  return dados_trainamento

def normalizar_dados(df):
    '''
    df - dataframe de valores a serem normalizados
    df_norm - dataframe normalizado pelo mínimo e máximo valor
    '''
    # Normalização dos dados
    x_float = df.astype('float')
    norm_min_max = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))
    df_norm = x_float.copy().apply(norm_min_max, axis=0)
    return df_norm

def get_atributos_treinamento():
    dados_treinamento = ler_base_treinamento()
    atributos_treinamento = dados_treinamento.drop(['target'], axis=1)
    return atributos_treinamento

def get_rotulos_treinamento():
    dados_treinamento = ler_base_treinamento()
    rotulos_treinamento = dados_treinamento['target']
    return rotulos_treinamento

def get_atributos_teste():
    dados_teste = ler_base_teste()
    atributos_teste = dados_teste.drop(['id'], axis=1)
    return atributos_teste


# Retorna uma amostra da base de dados 
def get_amostra_base(percentual=0.05):
    atributos_treinamento = get_atributos_treinamento()
    rotulos_treinamento = get_rotulos_treinamento()
    X_train, X_test, y_train, y_test = train_test_split(atributos_treinamento, rotulos_treinamento, train_size=percentual, stratify=rotulos_treinamento, random_state=2021)
    return [X_train, y_train]

def get_amostra_trat_val_ausentes():
    # Lê a base amostra de dados
    dados_treinamento, y = get_amostra_base(percentual=0.05)


    dados_faltantes = dados_treinamento[dados_treinamento == -1].any()
    dados_treinamento.values.tolist()
    missingValueColumns = dados_treinamento.columns[dados_faltantes].tolist()

    df_valores_nulos = dados_treinamento[missingValueColumns]

    qdt_valores_ausentes = df_valores_nulos[df_valores_nulos == -1	]
    qdt_valores_ausentes = (qdt_valores_ausentes == -1).sum(axis=0)

    qdt_valores_ausentes = qdt_valores_ausentes.to_frame()
    qdt_valores_ausentes = qdt_valores_ausentes.transpose()

    # Exclui os atributos com quantidade de valores ausentes maior que 2000
    dados_faltantes = qdt_valores_ausentes[qdt_valores_ausentes > 2000].any()

    # Lista de atributos que serão excluídos do conjunto de dados
    lista_atributos_excluir = qdt_valores_ausentes.columns[dados_faltantes].tolist()
    lista_atributos_excluir

    # Eclui os dados com muitos valores ausentes
    dados = dados_treinamento.drop(lista_atributos_excluir, axis=1)

    # Lista de atributos para substituir valores ausentes pela média
    lista_inserir_media = [x for x in missingValueColumns if x not in lista_atributos_excluir]

    # Substitui os valores ausentes pela média da respectiva coluna
    for coluna in lista_inserir_media:
        dados.loc[dados[coluna] == -1, coluna] = dados[coluna].mean()
    X = dados
    return [X, y]


