import pandas as pd
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import shapiro, levene, bartlett
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def main():
    data = []

    with open('sonarAndLLMShatteredpixeldungeon.json', 'r') as f:
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

    # Converter a lista em um DataFrame
    df = pd.DataFrame(data, columns=['score', 'code_smells', 'cognitive_complexity', 'comment_lines_density',
                                     'complexity', 'lines', 'sqale_rating', 'statements'])

    # Calcular o percentual de outliers para cada coluna
    outlier_percentages = {column: calculate_outliers(df, column) for column in df.columns}

    # Imprimir o percentual de outliers
    for column, percentage in outlier_percentages.items():
        print(f'Percentual de outliers para {column}: {percentage * 100:.2f}%')

    for column in df.columns:
        # Calcular e imprimir os valores
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median = df[column].median()
        mean = df[column].mean()
        std_dev = df[column].std()
        sum_val = df[column].sum()  # Calcular a soma dos valores
        print(f"Para a coluna {column}:")
        print(f"Soma {sum_val}:")
        plt.figure(figsize=(10, 6))
        plt.boxplot(df[column], vert=False)
        plt.title(f'Boxplot para a coluna {column}')
        plt.xlabel('Valores')
        plt.text(0.1, 0.2,f"Q1 (1º quartil): {Q1}\nMediana: {median}\nMédia: {mean}\nDesvio: {std_dev}\n"
                  f"Q3 (3º quartil): {Q3}\nIQR (Intervalo Interquartil): {IQR}\n"
                  f"Limite inferior para outliers: {lower_bound}\nLimite superior para outliers: {upper_bound}",
                  horizontalalignment='left', verticalalignment='center', fontsize=12, transform=plt.gca().transAxes)
        plt.savefig(f'graficos/boxplot_{column}.png', dpi=300)
        plt.close()
        # Criar gráfico de dispersão
        plt.figure(figsize=(12, 6))
        plt.scatter(df[column], df['score'])
        plt.title(f'Gráfico de dispersão para a coluna {column}')
        plt.xlabel(column)
        plt.ylabel('Score')
        # plt.show()
        plt.savefig(f'graficos/dispersao_{column}.png', dpi=300)
        plt.close()

    boxplot_group(df)
    print_occurrence_table(df)
    calcular_regressao_lienar_multipla(df)
    calcular_anova(df)
    calcular_estatisticas(df)
    calcular_correlacao(df, 'pearson')
    calcular_correlacao(df, 'spearman')
    calcular_significancia(df, 'score', 'code_smells')
    calcular_significancia(df, 'score', 'lines')
    calcular_significancia(df, 'score', 'complexity')
    calcular_significancia(df, 'score', 'cognitive_complexity')
    calcular_significancia(df, 'score', 'comment_lines_density')
    calcular_significancia(df, 'score', 'sqale_rating')
    calcular_significancia(df, 'score', 'statements')

def print_occurrence_table(df):
    # Create a pivot table with 'score' as index, 'sqale_rating' as columns, and count of occurrences as values
    pivot_table = pd.pivot_table(df, index='score', columns='sqale_rating', aggfunc='size', fill_value=0)

    # Print the pivot table
    print(pivot_table)

def calculate_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers) / len(df)

def calcular_regressao_lienar_multipla(df):
    # Adicione uma constante ao DataFrame para o intercepto
    df = sm.add_constant(df)

    # Ajuste o modelo
    model = sm.OLS(df['score'], df[['const', 'sqale_rating']]).fit()

    # Imprima o resumo
    print(model.summary())
    df['predicted_score'] = model.predict(df[['const', 'sqale_rating']])

    # Crie um gráfico de dispersão dos scores reais vs previstos
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sqale_rating', y='score', data=df, label='Real')
    sns.lineplot(x='sqale_rating', y='predicted_score', data=df, color='red', label='Previsto')
    plt.xlabel('Sqale Rating')
    plt.ylabel('Score')
    plt.title('Score Real vs Score Previsto')
    plt.legend()
    plt.savefig(f'graficos/regressao_linear_multipla.png', dpi=300)

def calcular_anova(df):
    model = ols('score ~ C(sqale_rating)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=1)
    print('Análise de correlação ANOVA')
    print(anova_table)
    # Aplicar o teste de Tukey HSD (Honest Significant Difference)
    tukey_results = pairwise_tukeyhsd(endog=df['score'], groups=df['sqale_rating'], alpha=0.05)

    # Imprimir os resultados
    print(tukey_results)

    # Perform Shapiro-Wilk test for normality
    for rating in df['sqale_rating'].unique():
        subset = df[df['sqale_rating'] == rating]['score']
        if subset.size >= 3:
            _, p_value = shapiro(subset)
            print(f"Shapiro-Wilk test for sqale_rating {rating}: p-value = {p_value}")
        else:
            print(
                f"Não é possível realizar o teste de Shapiro-Wilk para sqale_rating {rating} devido ao tamanho insuficiente")

    # Perform Levene's test for homogeneity of variances
    _, p_value = levene(*[df[df['sqale_rating'] == rating]['score'] for rating in df['sqale_rating'].unique()])
    print(f"Levene's test: p-value = {p_value}")

    # Perform Bartlett's test for homogeneity of variances
    _, p_value = bartlett(*[df[df['sqale_rating'] == rating]['score'] for rating in df['sqale_rating'].unique()])
    print(f"Bartlett's test: p-value = {p_value}")

    sns.boxplot(x='sqale_rating', y='score', data=df)

    # Defina o título e os rótulos do gráfico
    plt.title('Boxplot de Pontuações para Cada sqale_rating')
    plt.xlabel('sqale_rating')
    plt.ylabel('Pontuação')
    # Mostre o gráfico
    plt.savefig(f'graficos/boxplot_ratings.png', dpi=300)
    # plt.show()


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
            ax.set_ylim([0, 100])  # Limites do eixo y para 'score'
            # ax.set_yticks([-20, 0, 25, 50, 75, 100, 120])  # Definir ticks do eixo y
            # ax.set_yticklabels(['', '0', '25', '50', '75', '100', '']) # Definir rótulos do eixo y
        else:
            ax.set_ylim([4, 1])  # Limites do eixo y para 'sqale_rating'
            ax.set_yticks(range(0, 7))  # Definir ticks do eixo y
            ax.set_yticklabels(list(' ABCDE '))  # Definir rótulos do eixo y
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
    if p_value < 0.05:
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

