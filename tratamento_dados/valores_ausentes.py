from utils import db

def tratar_valores_ausentes(X):
    
    dados_faltantes = X[X == -1].any()
    X.values.tolist()
    missingValueColumns = X.columns[dados_faltantes].tolist()

    df_valores_nulos = X[missingValueColumns]

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
    resultado = X.drop(lista_atributos_excluir, axis=1)

    # Lista de atributos para substituir valores ausentes pela média
    lista_inserir_media = [x for x in missingValueColumns if x not in lista_atributos_excluir]

    # Substitui os valores ausentes pela média da respectiva coluna
    for coluna in lista_inserir_media:
        resultado.loc[resultado[coluna] == -1, coluna] = resultado[coluna].mean()
    
    return resultado


