import pandas as pd
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from scipy.stats import wilcoxon
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

def main():
    print(os.getcwd())

    # Defina as listas de dados e seus respectivos diretórios
    data_lists = {
        'clean_code': 'CleanCode',
        'bad_names_no_comments': 'BadNamesNoComments',
        'bad_names': 'BadNames',
        'no_comments': 'NoComments',
        'original': 'Original'
    }

    dfs = carregar_dataframes(data_lists)

    for nome_amigavel, nome_arquivo in data_lists.items():
        print(f"\nEstatisticas para {nome_amigavel}\n")
        calcular_estatisticas(dfs[nome_amigavel])

    print(dfs['original'])

    dfs['clean_code']['diferenca'] = ((dfs['clean_code']['score'] - dfs['original']['score']) / dfs['original']['score']) * 100
    dfs['clean_code']['alias'] = [1,3,4,5,2,6,7,9,8,10,11,12]
    print(f'diferenca_percentual clean_code:\n{dfs['clean_code'].sort_values(by='alias', ascending=True)}')

    dfs['bad_names']['diferenca'] = ((dfs['bad_names']['score'] - dfs['original']['score']) / dfs['original']['score']) * 100
    dfs['bad_names']['alias'] = [1, 3, 4, 5, 2, 6, 7, 9, 8, 10, 11, 12]
    print(f'diferenca_percentual bad_names:\n{dfs['bad_names'].sort_values(by='alias', ascending=True)}')

    dfs['no_comments']['diferenca'] = ((dfs['no_comments']['score'] - dfs['original']['score']) / dfs['original']['score']) * 100
    dfs['no_comments']['alias'] = [1, 3, 4, 5, 2, 6, 7, 9, 8, 10, 11, 12]
    print(f'diferenca_percentual no_comments:\n{dfs['no_comments'].sort_values(by='alias', ascending=True)}')

    # Gerar o gráfico de boxplot
    fig, ax = plt.subplots()
    ax.boxplot([dfs['original'].dropna(subset=['score'])['score'],
                dfs['no_comments'].dropna(subset=['score'])['score'],
                dfs['bad_names_no_comments'].dropna(subset=['score'])['score'],
                dfs['bad_names'].dropna(subset=['score'])['score'],
                dfs['clean_code'].dropna(subset=['score'])['score']])
    ax.set_ylim([0, 1])  # Ajuste da escala do eixo Y aqui
    ax.set_xticklabels(['Original', 'No Comments', 'No Comments Bad Names', 'Bad Names', 'Clean Code'], rotation=45)# Rótulos do eixo X

    plt.title('Boxplot dos Scores para readability Classifier')
    plt.tight_layout()
    plt.savefig('graficos/boxplot_all.png', dpi=300)  # Salvar com alta resolução


    # boxplot_group(df)


def carregar_dataframes(data_lists, path="."):
    dataframes = {}
    for nome_amigavel, nome_arquivo in data_lists.items():
        caminho_completo = f"{path}/{nome_arquivo}.csv"
        dataframes[nome_amigavel] = pd.read_csv(caminho_completo, names=['name', 'score'])
    return dataframes


def boxplot_group(df):
    # Lista de colunas
    columns = ['score', 'sqale_rating']
    # columns = ['statements', 'complexity', 'cognitive_complexity']
    # Dicionário de mapeamento para abreviar os rótulos
    abbreviations = {
        'code_smells': 'Code Smells',
        'statements': 'Statements',
        'cognitive_complexity': 'Cog Complx',
        'sqale_rating': 'Rating Sonar',
        'complexity': 'Complexity',
        'score': 'Score LLM'
    }
    # Dados para o boxplot
    data = [df[column] for column in columns]
    # Rótulos abreviados
    labels = [abbreviations[column] for column in columns]
    fig, axs = plt.subplots(1, 2, figsize=(4, 5))  # Subplots para cada variável
    for i, ax in enumerate(axs):
        ax.boxplot(data[i], vert=True)  # Adicionar coluna ao boxplot
        ax.set_title(labels[i])  # Título mais descritivo
        # ax.set_xlabel('Variáveis')  # Rótulo do eixo x mais descritivo
        if labels[i] == 'Score LLM':
            ax.set_ylim([40, 90])  # Limites do eixo y para 'score'
        else:
            ax.set_ylim([0, 5])  # Limites do eixo y para 'sqale_rating'
            ax.set_yticks(range(1, 6))  # Definir ticks do eixo y
            ax.set_yticklabels(list('ABCDE'))  # Definir rótulos do eixo y
    # plt.title('Boxplots das variáveis do estudo')  # Título mais descritivo
    # plt.xlabel('Variáveis')  # Rótulo do eixo x mais descritivo
    plt.tight_layout()
    plt.savefig('graficos/boxplot_all.png', dpi=300)  # Salvar com alta resolução
    plt.close()

def calculate_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers) / len(df)

def calcular_estatisticas(df):
    # Calcular o percentual de outliers para cada coluna
    outlier_percentages = {column: calculate_outliers(df, column) for column in df.columns if column == 'score'}

    # Imprimir o percentual de outliers
    for column, percentage in outlier_percentages.items():
        print(f'Percentual de outliers para {column}: {percentage * 100:.2f}%')

    # Calcular a média, mediana e desvio padrão de cada variável
    statistics = {column: {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std_dev': df[column].std(),
        'skewness': df[column].skew(),
        'modes': df[column].mode().values[0],
        'std_error': df[column].std() / np.sqrt(len(df))
    } for column in df.columns if column == 'score'}

    for column, stats in statistics.items():
        print(f"Estatísticas para a coluna '{column}':")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value}")
        print()

if __name__ == "__main__":
    main()

# {"score":"60","reasoning":"The code is very simple and easy to read, but it lacks context and functionality.
# It seems to be an incomplete class file, which makes it difficult to evaluate its overall quality.","tokens":219,
# "sonarData":{"component":{"id":"AYvyvWT8pBbp5z45sMS-","key":"quarkusio_quarkus:core/junit4-mock/src/main/java/org/junit/rules/ExternalResource.java","name":"ExternalResource.java","qualifier":"FIL","path":"core/junit4-mock/src/main/java/org/junit/rules/ExternalResource.java","language":"java",
# "measures":[{"metric":"complexity","value":"1","bestValue":false},
# {"metric":"code_smells","value":"1","bestValue":false},
# {"metric":"cognitive_complexity","value":"0","bestValue":true},
# {"metric":"files","value":"1","bestValue":false},
# {"metric":"comment_lines_density","value":"0.0","bestValue":false},
# {"metric":"lines","value":"8","bestValue":false},
# {"metric":"sqale_rating","value":"1.0","bestValue":true}]}}}

