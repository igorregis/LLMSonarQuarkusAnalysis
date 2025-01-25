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
    data = []
    print(os.getcwd())

    loadData(data, 'controlledOriginalGemini15pro-1.json')
    loadData(data, 'controlledOriginalGemini15pro-2.json')
    loadData(data, 'controlledOriginalGemini15pro-3.json')
    loadData(data, 'controlledOriginalGemini15pro-4.json')
    loadData(data, 'controlledOriginalGemini15pro-5.json')
    loadData(data, 'controlledOriginalGemini15pro-6.json')
    loadData(data, 'controlledOriginalGemini15pro-7.json')
    loadData(data, 'controlledOriginalGemini15pro-8.json')
    loadData(data, 'controlledOriginalGemini15pro-9.json')
    loadData(data, 'controlledOriginalGemini15pro-10.json')

    # Converter a lista em um DataFrame
    df = pd.DataFrame(data, columns=['name', 'score', 'code_smells', 'cognitive_complexity', 'comment_lines_density',
                                     'complexity', 'lines', 'sqale_rating', 'statements'])


    calcular_estatisticas(df)
    calcular_variacao(df)


def calcular_variacao(df):
    # Suponha que 'df' é o seu DataFrame e que 'name' e 'score' são colunas no DataFrame
    df['score'] = df['score'].astype(int)  # Certifique-se de que a coluna 'score' é de tipo int
    # Agrupe por 'name' e calcule a variação (max - min) para 'score'
    score_variation = df.groupby('name')['score'].apply(lambda x: x.max() - x.min())
    print(score_variation)
    # Para cada 'name', calcule o valor-p do Teste de Wilcoxon
    for name, group in df.groupby('name'):
        scores = group['score'].values
        if np.std(scores) != 0:  # Se não houver variação nos scores, o teste não pode ser aplicado
            w, p = wilcoxon(scores - np.median(scores))
            print(f"O valor-p para o Teste de Wilcoxon para {name} é {p}.")
    # Para cada 'name', calcule o percentual de registros com valores diferentes
    comulative_percentage = 0
    for name, group in df.groupby('name'):
        scores = group['score'].values
        unique_scores = np.unique(scores)
        if len(unique_scores) == 1:  # Se todos os scores são iguais
            percentage = 0
        else:
            percentage = (len(unique_scores)-1) / len(scores) * 100
        print(f"{percentage}% de registros com valores diferentes para {name}")
        comulative_percentage += percentage
    print(f"{comulative_percentage/1200*100}% de variação total")

def loadData(data, file):
    with open(file, 'r') as f:
        for line in f:
            json_line = json.loads(line)

            # Extrair os valores necessários
            score = int(json_line['score'])
            name = json_line['sonarData']['component'].get('name')
            measures = {m['metric']: float(m['value']) for m in json_line['sonarData']['component']['measures']}
            code_smells = measures.get('code_smells', 0)
            cognitive_complexity = measures.get('cognitive_complexity', 0)
            complexity = measures.get('complexity', 0)
            lines = measures.get('lines', 0)
            sqale_rating = measures.get('sqale_rating', 0)
            statements = measures.get('statements', 0)
            comment_lines_density = measures.get('comment_lines_density', 0)

            # Adicionar ao conjunto de dados
            data.append([name, score, code_smells, cognitive_complexity, comment_lines_density,
                         complexity, lines, sqale_rating, statements])

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

