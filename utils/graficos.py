import matplotlib.pyplot as plt
import seaborn as sns
from utils import db

def grafico_balanceamento_classes():
    dados_treinamento = db.ler_base_treinamento()
    plt.figure(figsize=(15,8))

    # faz um agrupamento da coluna 'target'
    ax = sns.countplot('target', data=dados_treinamento)
    ax.xlabel = 'dsadsadas'
    for p in ax.patches:
        ax.annotate('{:.2f}%'.format(100*p.get_height()/len(dados_treinamento['target'])), 
            (p.get_x() + 0.3, p.get_height() + 10000))