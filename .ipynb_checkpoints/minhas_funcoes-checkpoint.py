import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def get_description(df, quant_col):
    """Recebe um dataframe e uma lista de colunas. Retorna barplots que descrevem a distribuição destas variáveis"""
    for i in range(0, len(quant_col)):
        sns.countplot(data=df, x=quant_col[i])
        plt.show()



def get_biv(df, variaveis, alvo, tipo_var):
    """Recebe um df, uma lista de variaveis, uma variavel alvo, um tipo de variavel, e retorna uma analise bivariada das variaveis em questão
    com a variável alvo"""
    if tipo_var=='quantitativa':
        for i in range(0, len(variaveis)):
            sns.boxplot(data=df, x=alvo, y=variaveis[i])
            plt.show()
    elif tipo_var=='qualitativa':
        for i in range(0, len(variaveis)):
            sns.countplot(data=df, x=variaveis[i], hue=alvo)
            plt.show()


def nan_to_0(serie):
    """Recebe uma série do pandas. Retorna a mesma série com os seus valores não numéricos trocados por 0"""
    return pd.to_numeric(serie, errors='coerce').fillna(0)


