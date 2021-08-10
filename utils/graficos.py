import matplotlib.pyplot as plt
import seaborn as sns
from utils import db
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def grafico_balanceamento_classes():
    dados_treinamento = db.ler_base_treinamento()
    plt.figure(figsize=(15,8))

    # faz um agrupamento da coluna 'target'
    ax = sns.countplot('target', data=dados_treinamento)
    ax.xlabel = 'dsadsadas'
    for p in ax.patches:
        ax.annotate('{:.2f}%'.format(100*p.get_height()/len(dados_treinamento['target'])), 
            (p.get_x() + 0.3, p.get_height() + 10000))



def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i]+5,y[i], ha = 'center')

def pandas_bar_plot(y, color_opt='grayscale'):
    cores = ['#000000', '#444444', '#555555', '#666666', '#777777', '#888888', '#999999', '#aaaaaa', '#bbbbbb', '#cccccc']
    if(color_opt == 'grayscale'):
        cores = ['#000000', '#444444', '#555555', '#666666', '#777777', '#888888', '#999999', '#aaaaaa', '#bbbbbb', '#cccccc']
    
    frequencias_classes = np.array([(y == idx).sum() for idx in [0, 1]])
    height = frequencias_classes
    bars = [0, 1]
    y_pos = np.arange(len(bars))
    fig, ax = plt.subplots()
    plt.title('Quantidade de instâncias por classe')
    plt.xlabel('Classes')
    plt.ylabel('Quantidade')
    plt.grid(False)
    # Create bars
    
    plt.bar(y_pos, height, width=.5)
    addlabels(y_pos, height)

    # Create names on the x-axis
    plt.xticks(y_pos, bars)

    # Show graphic
    plt.show()

def grafico_missing_values(percentual=0.1):
    dados_treinamento, y = db.get_amostra_base(percentual=percentual)
    

    dados_faltantes = dados_treinamento[dados_treinamento == -1].any()
    dados_treinamento.values.tolist()
    missingValueColumns = dados_treinamento.columns[dados_faltantes].tolist()
    
    plt.rcParams["figure.figsize"] = [10, 8]
    plt.rcParams["figure.autolayout"] = True

    dados_faltantes = dados_treinamento[dados_treinamento == -1].any()

    missingValueColumns = dados_treinamento.columns[dados_faltantes].tolist()
    missingValueColumns
    df_null = dados_treinamento[missingValueColumns]

   
    df_null[df_null == (-1)].sum(axis=0)


    plt.rcParams["figure.figsize"] = [10, 8]
    plt.rcParams["figure.autolayout"] = True

    count_missing_values = df_null[df_null == -1	]
    count_missing_values = (count_missing_values == -1).sum(axis=0)
    print(type(count_missing_values))
    count_missing_values = count_missing_values.to_frame()
    

    ax = count_missing_values.plot(kind='bar', legend=False, rot=45, linewidth=50)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.005))

