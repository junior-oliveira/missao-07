import pandas as pd
import numpy as np

def ler_base_treinamento():
  dados_treinamento = pd.read_csv('bases_de_dados/train.csv')
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
    atributos_treinamento = dados_treinamento.drop(['target', 'id'], axis=1)
    return atributos_treinamento

def get_rotulos_treinamento():
    dados_treinamento = ler_base_treinamento()
    rotulos_treinamento = dados_treinamento['target']
    return rotulos_treinamento

def get_atributos_teste():
    dados_teste = ler_base_teste()
    atributos_teste = dados_teste.drop(['id'], axis=1)
    return atributos_teste

