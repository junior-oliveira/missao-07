
import numpy as np

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