import pandas as pd
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

def main():
    data = []
    print(os.getcwd())

    loadData(data, 'controlledNoCommentsGPT35.json')
    loadData(data, 'sonarAndLLM-2.json')
    loadData(data, 'sonarAndLLM-3.json')
    loadData(data, 'sonarAndLLM-4.json')
    loadData(data, 'sonarAndLLM-5.json')
    loadData(data, 'sonarAndLLM-6.json')
    loadData(data, 'sonarAndLLM-7.json')
    loadData(data, 'sonarAndLLM-8.json')
    loadData(data, 'sonarAndLLM-9.json')
    loadData(data, 'sonarAndLLM-10.json')

    # Converter a lista em um DataFrame
    df = pd.DataFrame(data, columns=['score', 'code_smells', 'cognitive_complexity', 'comment_lines_density',
                                     'complexity', 'lines', 'sqale_rating', 'statements'])

    # Calcular o percentual de outliers para cada coluna
    outlier_percentages = {column: calculate_outliers(df, column) for column in df.columns}

    # Imprimir o percentual de outliers
    for column, percentage in outlier_percentages.items():
        print(f'Percentual de outliers para {column}: {percentage * 100:.2f}%')

    calcular_estatisticas(df)



def loadData(data, file):
    with open(file, 'r') as f:
        for line in f:
            json_line = json.loads(line)

            # Extrair os valores necessários
            score = int(json_line['score'])
            measures = {m['metric']: float(m['value']) for m in json_line['sonarData']['component']['measures']}
            code_smells = measures.get('code_smells', 0)
            cognitive_complexity = measures.get('cognitive_complexity', 0)
            complexity = measures.get('complexity', 0)
            lines = measures.get('lines', 0)
            sqale_rating = measures.get('sqale_rating', 0)
            statements = measures.get('statements', 0)
            comment_lines_density = measures.get('comment_lines_density', 0)

            # Adicionar ao conjunto de dados
            data.append([score, code_smells, cognitive_complexity, comment_lines_density,
                         complexity, lines, sqale_rating, statements])


def calculate_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers) / len(df)

def calcular_anova(df):
    model = ols('score ~ C(sqale_rating)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=1)
    print('Análise de correlação ANOVA')
    print(anova_table)


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


def calcular_estatisticas(df):
    # Calcular a média, mediana e desvio padrão de cada variável
    mean = df.mean()
    median = df.median()
    std_dev = df.std()
    skewness = df.skew()
    modes = df.mode().loc[0]
    std_error = std_dev / np.sqrt(len(df))

    print("\nMédia:\n", mean)
    print("\nMediana:\n", median)
    print("\nDesvio Padrão:\n", std_dev)
    print("\nCoeficiente de Assimetria:\n", skewness)
    print("\nModa:\n", modes)
    print("\nErro Padrão:\n", std_error)


def calcular_correlacao(df, method='pearson'):
    # Calcular a correlação
    correlation = df.corr(method=method)
    print(f"\nCorrelação de {method.capitalize()}:\n", correlation)


def calcular_significancia(df, col1, col2):
    # Calcule a correlação e o valor-p
    print(f'\nSignificância de {col2}')
    correlation, p_value = stats.pearsonr(df[col1], df[col2])
    correlation_spearman, p_value_spearman = stats.spearmanr(df[col1], df[col2])

    print(f'Correlação Pearson: {correlation}')
    print(f'Valor-p Pearson: {p_value}')
    print(f'Correlação Spearman: {correlation_spearman}')
    print(f'Valor-p Spearman: {p_value_spearman}')

    # Verifique se a correlação é estatisticamente significativa
    if p_value_spearman < 0.05:
        print('A correlação é estatisticamente significativa.')
    else:
        print('A correlação não é estatisticamente significativa.')


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

