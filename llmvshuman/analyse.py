import krippendorff
import re
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import squareform, pdist
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import json
import pandas as pd
import os
from scipy.stats import spearmanr
from itertools import combinations
from matplotlib.ticker import FuncFormatter
from factor_analyzer import FactorAnalyzer
import warnings
import pingouin as pg
from numpy.ma.extras import column_stack

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.*"
)

def carregar_dados_csv(caminho_arquivo, codificacao='utf-8'):
        df = pd.read_csv(caminho_arquivo, encoding=codificacao)
        return df

def dividir_dataframe(df):
    if df.shape[1] <= 14:
        df1 = df
        df2 = df[['Id']] #cria um dataframe apenas com a coluna id.
        return df1, df2

    df1 = pd.concat([df['Id'], df.iloc[:, 1:14]], axis=1)  # Inclui 'Id' e as 13 colunas seguintes
    df1 = df1.drop(['Email', 'Name', 'Language', 'Start time', 'Completion time'], axis=1)

    df2 = df[['Id']].join(df.iloc[:, 14:])  # Inclui 'Id' e as colunas restantes

    return df1, df2

def dividir_dataframe_por_pergunta(df, pergunta, resposta):
    if pergunta not in df.columns:
        raise ValueError(f"A coluna {pergunta} não está presente no DataFrame.")

    df_nan = df[~df[pergunta].astype(str).str.contains(resposta, na=False)]
    df_not_nan = df[df[pergunta].astype(str).str.contains(resposta, na=False)]

    return df_nan, df_not_nan


def load_and_clean_data():
    dados = carregar_dados_csv('Humans/raw-data.csv', 'ISO-8859-1')
    df_demografia, df_respostas = dividir_dataframe(dados)
    df_demografia['Qual opção melhor descreve seu cargo atual?'] = df_demografia['Qual opção melhor descreve seu cargo atual?'].replace('Senior ou equivalente', 'Senior')
    df_demografia['Qual opção melhor descreve seu cargo atual?'] = df_demografia['Qual opção melhor descreve seu cargo atual?'].replace('Pleno ou equivalente', 'Mid')
    df_demografia['Qual opção melhor descreve seu cargo atual?'] = df_demografia['Qual opção melhor descreve seu cargo atual?'].replace('Junior ou equivalente', 'Junior')
    df_demografia['Qual opção melhor descreve seu cargo atual?'] = df_demografia['Qual opção melhor descreve seu cargo atual?'].replace('Staff ou equivalente', 'Staff')
    df_demografia['Qual opção melhor descreve seu cargo atual?'] = df_demografia['Qual opção melhor descreve seu cargo atual?'].replace('Principal ou equivalente', 'Principal')
    df_demografia['Qual seu cargo?'] = df_demografia['Qual seu cargo?'].replace('Assessor de TI I', 'Senior')
    df_demografia['Qual seu cargo?'] = df_demografia['Qual seu cargo?'].replace('Assessor de TI II', 'Mid')
    df_demografia['Qual seu cargo?'] = df_demografia['Qual seu cargo?'].replace('Assessor de TI III ou III Trainee ou Escriturário', 'Junior')
    df_demografia['Qual seu cargo?'] = df_demografia['Qual seu cargo?'].replace('Gerente de Equipe / Tech Lead ou Especialista III', 'Staff')
    df_demografia['Qual seu cargo?'] = df_demografia['Qual seu cargo?'].replace('Gerente de Soluções ou Especialista II', 'Principal')
    df_mercado, df_bb = dividir_dataframe_por_pergunta(df_demografia, 'Você é funcionário da TI do Banco do Brasil?',
                                                       'Sim')
    indices_a_deletar = [5, 6, 7, 8]
    colunas_a_deletar = df_mercado.columns[indices_a_deletar]
    df_mercado = df_mercado.drop(colunas_a_deletar, axis=1)
    df_mercado, df_terceiro = dividir_dataframe_por_pergunta(df_mercado, 'Você é funcionário da TI do Banco do Brasil?',
                                                             'terceirizado')
    indices_a_deletar = [1, 2, 3, 4]
    colunas_a_deletar = df_bb.columns[indices_a_deletar]
    df_bb = df_bb.drop(colunas_a_deletar, axis=1)
    df_mercado = df_mercado[df_mercado['Há quanto tempo você programa?'] != 'Não programo']
    df_terceiro = df_terceiro[df_terceiro['Há quanto tempo você programa?'] != 'Não programo']
    df_bb = df_bb[df_bb['Há quanto tempo você programa?1'] != 'Não programo']

    novas_colunas = {'Você é funcionário da TI do Banco do Brasil?': 'terceiro', 'Há quanto tempo você programa?': 'experience', 'Qual das linguagens abaixo você considera que conhece melhor?': 'language', 'Qual opção melhor descreve seu cargo atual?':'role'}
    df_mercado = df_mercado.rename(columns=novas_colunas)
    df_mercado['terceiro'] = df_mercado['terceiro'].replace('Não (não trabalho para a TI do BB)', 'No')

    novas_colunas = {'Você é funcionário da TI do Banco do Brasil?': 'terceiro', 'Há quanto tempo você programa?': 'experience', 'Qual das linguagens abaixo você considera que conhece melhor?': 'language', 'Qual opção melhor descreve seu cargo atual?':'role'}
    df_terceiro = df_terceiro.rename(columns=novas_colunas)
    df_terceiro['terceiro'] = df_terceiro['terceiro'].replace('Não (sou terceirizado)', 'Yes')
    df_terceiro['terceiro'] = df_terceiro['terceiro'].replace('Sou terceirizado no BB', 'Yes')

    novas_colunas = {'Qual seu cargo?': 'role', 'Qual das linguagens abaixo você considera que conhece melhor?1': 'language', 'Há quanto tempo você programa?1': 'experience', 'Em que ano você entrou no BB?':'job time'}
    df_bb = df_bb.rename(columns=novas_colunas)

    df_respostas.columns.values[1] = 'Original-MonthPlus.java-score'
    df_respostas.columns.values[2] = 'Original-MonthPlus.java-reasoning'
    df_respostas.columns.values[3] = 'BadNames-Combine.java-score'
    df_respostas.columns.values[4] = 'BadNames-Combine.java-reasoning'
    df_respostas.columns.values[5] = 'NoComments-MonthSubtrai.java-score'
    df_respostas.columns.values[6] = 'NoComments-MonthSubtrai.java-reasoning'
    df_respostas.columns.values[7] = 'Original-MonthPrimeiroMesDoTrimestre.java-score'
    df_respostas.columns.values[8] = 'Original-MonthPrimeiroMesDoTrimestre.java-reasoning'
    df_respostas.columns.values[9] = 'BadNames-GetMedia.java-score'
    df_respostas.columns.values[10] = 'BadNames-GetMedia.java-reasoning'
    df_respostas.columns.values[11] = 'NoComments-CarregarFilhos.java-score'
    df_respostas.columns.values[12] = 'NoComments-CarregarFilhos.java-reasoning'
    df_respostas.columns.values[13] = 'Original-CalculaDVBase10.java-score'
    df_respostas.columns.values[14] = 'Original-CalculaDVBase10.java-reasoning'
    df_respostas.columns.values[15] = 'BadNames-CalculaAreaCirculo.java-score'
    df_respostas.columns.values[16] = 'BadNames-CalculaAreaCirculo.java-reasoning'
    df_respostas.columns.values[17] = 'NoComments-CalculaAreaTrianguloIsoceles.java-score'
    df_respostas.columns.values[18] = 'NoComments-CalculaAreaTrianguloIsoceles.java-reasoning'
    df_respostas.columns.values[19] = 'Original-AnoBissexto.java-score'
    df_respostas.columns.values[20] = 'Original-AnoBissexto.java-reasoning'
    df_respostas.columns.values[21] = 'NoComments-MonthPlus.java-score'
    df_respostas.columns.values[22] = 'NoComments-MonthPlus.java-reasoning'
    df_respostas.columns.values[23] = 'Original-Combine.java-score'
    df_respostas.columns.values[24] = 'Original-Combine.java-reasoning'
    df_respostas.columns.values[25] = 'BadNames-MonthSubtrai.java-score'
    df_respostas.columns.values[26] = 'BadNames-MonthSubtrai.java-reasoning'
    df_respostas.columns.values[27] = 'NoComments-MonthPrimeiroMesDoTrimestre.java-score'
    df_respostas.columns.values[28] = 'NoComments-MonthPrimeiroMesDoTrimestre.java-reasoning'
    df_respostas.columns.values[29] = 'Original-GetMedia.java-score'
    df_respostas.columns.values[30] = 'Original-GetMedia.java-reasoning'
    df_respostas.columns.values[31] = 'BadNames-CarregarFilhos.java-score'
    df_respostas.columns.values[32] = 'BadNames-CarregarFilhos.java-reasoning'
    df_respostas.columns.values[33] = 'NoComments-CalculaDVBase10.java-score'
    df_respostas.columns.values[34] = 'NoComments-CalculaDVBase10.java-reasoning'
    df_respostas.columns.values[35] = 'Original-CalculaAreaCirculo.java-score'
    df_respostas.columns.values[36] = 'Original-CalculaAreaCirculo.java-reasoning'
    df_respostas.columns.values[37] = 'BadNames-CalculaAreaTrianguloIsoceles.java-score'
    df_respostas.columns.values[38] = 'BadNames-CalculaAreaTrianguloIsoceles.java-reasoning'
    df_respostas.columns.values[39] = 'NoComments-AnoBissexto.java-score'
    df_respostas.columns.values[40] = 'NoComments-AnoBissexto.java-reasoning'
    df_respostas.columns.values[41] = 'BadNames-MonthPlus.java-score'
    df_respostas.columns.values[42] = 'BadNames-MonthPlus.java-reasoning'
    df_respostas.columns.values[43] = 'NoComments-Combine.java-score'
    df_respostas.columns.values[44] = 'NoComments-Combine.java-reasoning'
    df_respostas.columns.values[45] = 'Original-MonthSubtrai.java-score'
    df_respostas.columns.values[46] = 'Original-MonthSubtrai.java-reasoning'
    df_respostas.columns.values[47] = 'BadNames-MonthPrimeiroMesDoTrimestre.java-score'
    df_respostas.columns.values[48] = 'BadNames-MonthPrimeiroMesDoTrimestre.java-reasoning'
    df_respostas.columns.values[49] = 'NoComments-GetMedia.java-score'
    df_respostas.columns.values[50] = 'NoComments-GetMedia.java-reasoning'
    df_respostas.columns.values[51] = 'Original-CarregarFilhos.java-score'
    df_respostas.columns.values[52] = 'Original-CarregarFilhos.java-reasoning'
    df_respostas.columns.values[53] = 'BadNames-CalculaDVBase10.java-score'
    df_respostas.columns.values[54] = 'BadNames-CalculaDVBase10.java-reasoning'
    df_respostas.columns.values[55] = 'NoComments-CalculaAreaCirculo.java-score'
    df_respostas.columns.values[56] = 'NoComments-CalculaAreaCirculo.java-reasoning'
    df_respostas.columns.values[57] = 'Original-CalculaAreaTrianguloIsoceles.java-score'
    df_respostas.columns.values[58] = 'Original-CalculaAreaTrianguloIsoceles.java-reasoning'
    df_respostas.columns.values[59] = 'BadNames-AnoBissexto.java-score'
    df_respostas.columns.values[60] = 'BadNames-AnoBissexto.java-reasoning'
    df_respostas.columns.values[61] = 'Comments'

    df_notas = df_respostas.filter(like='score').copy()
    df_notas['Id'] = df_respostas['Id'].values
    ultima_coluna = df_notas.columns[-1]
    df_notas = df_notas[[ultima_coluna] + df_notas.columns[:-1].tolist()]

    return df_notas, df_respostas, df_mercado, df_bb, df_terceiro

# Esta função separa os respondentes de acordo com as questões que eles responderam. Isso nos permitirá depois unir as respostas se todos para uma análise
def split_random_groups(df):
    df1 = df[df['Original-MonthPlus.java-score'].notna()]
    df2 = df[df['BadNames-MonthPlus.java-score'].notna()]
    df3 = df[df['NoComments-MonthPlus.java-score'].notna()]

    df1 = df1.dropna(axis=1, how='all')
    df2 = df2.dropna(axis=1, how='all')
    df3 = df3.dropna(axis=1, how='all')

    return df1, df2, df3

def load(scenario, llm):
    all_files = os.listdir(f'./{llm}/{scenario}')
    files = [file for file in all_files if file.startswith('llmvshuman'+scenario+llm)]
    files.sort()
    counter = 1
    temp = pd.DataFrame()
    for file in files:
        loaded_df = load_json_data(f'./{llm}/{scenario}/{file}', f'{llm}-{counter}')
        loaded_df.index = scenario + '-' +  loaded_df.index + '-score'
        if temp.empty:
            temp = loaded_df.copy()
        else:
            # temp = pd.concat([temp, loaded_df])
            temp = loaded_df.join(temp)
        counter = counter+1
    return temp

def load_json_data(file_path, column):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                json_obj = json.loads(line)
                data.append({'name': json_obj.get('name'), column: json_obj.get('score')})
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
    df = pd.DataFrame(data).set_index('name').sort_values(by='name')
    df[column] = df[column].astype(int)
    return df.copy()

def calcular_correlacoes_spearman_all(medias, coluna_x):
    correlacoes = {}
    chaves = list(medias.keys())
    # Itera sobre todas as combinações de pares de DataFrames
    for chave1, chave2 in combinations(chaves, 2):
        df1 = medias[chave1]
        df2 = medias[chave2]
        # Verifica se a coluna existe em ambos os DataFrames
        if coluna_x in df1.columns and coluna_x in df2.columns:
            # Calcula a correlação de Spearman e o p-valor
            # print(f'Correlação entre {chave1} e {chave2}:\n{df1[coluna_x]}\n{df2[coluna_x]}')
            correlacao, p_valor = spearmanr(df1[coluna_x], df2[coluna_x], nan_policy='omit')
            # print('correlacao:', correlacao, 'p_valor:', p_valor)
            correlacoes[(chave1, chave2)] = {'correlacao': correlacao, 'p_valor': p_valor}
        else:
            correlacoes[(chave1, chave2)] = {'correlacao': None, 'p_valor': None, 'erro': f"Coluna '{coluna_x}' não encontrada em um dos DataFrames."}

    return correlacoes


def calcular_correlacoes_spearman(medias, coluna_x):
    correlacoes = {}
    chaves = list(medias.keys())
    # Itera sobre todas as combinações de pares de DataFrames
    for chave1, chave2 in combinations(chaves, 2):
        # Verifica se as chaves compartilham os últimos 9 caracteres
        if chave1[-8:] == chave2[-8:]:
            df1 = medias[chave1]
            df2 = medias[chave2]

            # Verifica se a coluna existe em ambos os DataFrames
            if coluna_x in df1.columns and coluna_x in df2.columns:
                # Calcula a correlação de Spearman e o p-valor
                # print(f'Correlação entre {chave1} e {chave2}:\n{df1[coluna_x]}\n{df2[coluna_x]}')
                correlacao, p_valor = spearmanr(df1[coluna_x], df2[coluna_x], nan_policy='omit')
                # print('correlacao:', correlacao, 'p_valor:', p_valor)
                correlacoes[(chave1, chave2)] = {'correlacao': correlacao, 'p_valor': p_valor}
            else:
                correlacoes[(chave1, chave2)] = {'correlacao': None, 'p_valor': None, 'erro': f"Coluna '{coluna_x}' não encontrada em um dos DataFrames."}

    return correlacoes

def calcular_teste_wilcoxon_all(medias, coluna_x):
    resultados_wilcoxon = {}
    chaves = list(medias.keys())

    # Itera sobre todas as combinações de pares de DataFrames
    for chave1, chave2 in combinations(chaves, 2):
        df1 = medias[chave1]
        df2 = medias[chave2]

        # Verifica se a coluna existe em ambos os DataFrames
        if coluna_x in df1.columns and coluna_x in df2.columns:
            # Remove valores NaN para o teste de Wilcoxon
            valores1 = df1[coluna_x].dropna()
            valores2 = df2[coluna_x].dropna()

            # Garante que há dados suficientes para realizar o teste
            if len(valores1) > 0 and len(valores2) > 0:
                # Calcula o teste de Wilcoxon
                estatistica, p_valor = wilcoxon(valores1, valores2)
                resultados_wilcoxon[(chave1, chave2)] = {'estatistica': estatistica, 'p_valor': p_valor}
            else:
                resultados_wilcoxon[(chave1, chave2)] = {'estatistica': None, 'p_valor': None, 'erro': "Dados insuficientes para realizar o teste."}
        else:
            resultados_wilcoxon[(chave1, chave2)] = {'estatistica': None, 'p_valor': None, 'erro': f"Coluna '{coluna_x}' não encontrada em um dos DataFrames."}

    return resultados_wilcoxon

def calcular_teste_wilcoxon(medias, coluna_x):
    resultados_wilcoxon = {}
    chaves = list(medias.keys())

    # Itera sobre todas as combinações de pares de DataFrames
    for chave1, chave2 in combinations(chaves, 2):
        # Verifica se as chaves compartilham os últimos 9 caracteres
        if chave1[-8:] == chave2[-8:]:
            df1 = medias[chave1]
            df2 = medias[chave2]

            # Verifica se a coluna existe em ambos os DataFrames
            if coluna_x in df1.columns and coluna_x in df2.columns:
                # Remove valores NaN para o teste de Wilcoxon
                valores1 = df1[coluna_x].dropna()
                valores2 = df2[coluna_x].dropna()

                # Garante que há dados suficientes para realizar o teste
                if len(valores1) > 0 and len(valores2) > 0:
                    # Calcula o teste de Wilcoxon
                    estatistica, p_valor = wilcoxon(valores1, valores2)
                    resultados_wilcoxon[(chave1, chave2)] = {'estatistica': estatistica, 'p_valor': p_valor}
                else:
                    resultados_wilcoxon[(chave1, chave2)] = {'estatistica': None, 'p_valor': None, 'erro': "Dados insuficientes para realizar o teste."}
            else:
                resultados_wilcoxon[(chave1, chave2)] = {'estatistica': None, 'p_valor': None, 'erro': f"Coluna '{coluna_x}' não encontrada em um dos DataFrames."}

    return resultados_wilcoxon

def count_scores(df):
    resultado = pd.DataFrame()
    sorted_columns = sorted(df.columns)
    for column in sorted_columns:
        if column != 'Id':  # Skip the 'Id' column
            valid_scores = df[column].dropna()  # Remove NaN values
            scores_above_or_equal_5 = valid_scores[valid_scores >= 5]
            scores_below_5 = valid_scores[valid_scores < 5]

            count_above_or_equal_5 = len(scores_above_or_equal_5)
            count_below_5 = len(scores_below_5)
            difference = count_above_or_equal_5 - count_below_5
            if -5 <= difference <= 5:
                veredicto = "INDEFINIDO"
            elif difference > 5:
                veredicto = "LEGÍVEL"
            else:
                veredicto = "ILEGÍVEL"
            # print(f"Column: {column} Scores >= 5: {count_above_or_equal_5} Scores < 5: {count_below_5} - {veredicto}")
            resultado[column] = veredicto
    return resultado

def plot_score_distribution(df):
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Melt the dataframe to convert it from wide to long format
    df_melted = pd.melt(df_copy, value_vars=df_copy.columns[1:], var_name='column', value_name='score')
    # Drop NaN values
    df_valid_scores = df_melted.dropna(subset=['score'])
    # print(df_valid_scores.to_string())

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df_valid_scores['score'], kde=True, bins=11, binrange=(0, 10))  # Use histplot for histogram and KDE
    plt.title('Distribution of Valid Scores')
    plt.xlabel('Score')
    plt.xlim(0, 10)
    plt.xticks(np.arange(0, 11, 1))
    plt.ylabel('Frequency')

    # Create the 'graficos' directory if it doesn't exist
    if not os.path.exists('graficos'):
        os.makedirs('graficos')

    # Save the plot to 'graficos/distribuicao.png'
    plt.savefig('graficos/distribuicao.png', dpi=300)
    plt.close()

def criar_heatmap_correlacoes(scenario, correlacoes):
    # Extrai os nomes únicos dos DataFrames
    nomes_unicos = sorted(list(set([nome for par in correlacoes.keys() for nome in par])))

    # Cria uma matriz para armazenar os valores de correlação
    matriz_correlacao = pd.DataFrame(index=nomes_unicos, columns=nomes_unicos)

    # Preenche a matriz com os valores de correlação
    for (nome1, nome2), valores in correlacoes.items():
        if valores['correlacao'] is not None:
            if valores['p_valor'] is not None and valores['p_valor'] <= 0.05:
                matriz_correlacao.loc[nome1, nome2] = valores['correlacao']
                matriz_correlacao.loc[nome2, nome1] = valores['correlacao']  # Correlação é simétrica
            else:
                matriz_correlacao.loc[nome1, nome2] = 0  # Define como 0 se não for significativo
                matriz_correlacao.loc[nome2, nome1] = 0
        else:
            matriz_correlacao.loc[nome1, nome2] = np.nan
            matriz_correlacao.loc[nome2, nome1] = np.nan

    # Converte todos os valores para numéricos (float) e substitui valores não numéricos por NaN
    matriz_correlacao = matriz_correlacao.apply(pd.to_numeric, errors='coerce')
    matriz_correlacao = matriz_correlacao.rename(index=lambda x: x.replace(scenario, ''), columns=lambda x: x.replace(scenario, ''))
    # Gera o heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Heatmap de Correlações de Spearman ({scenario})')
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=35, va='top')
    plt.savefig(f'graficos/heatmap_{scenario}.png', dpi=300)
    plt.close()

def criar_graficos_dispersao(scenario, dataframes):
    # Obtém a lista de nomes dos arquivos Java
    nomes_arquivos = dataframes[list(dataframes.keys())[0]].index
    # Remove a substring do cenário dos rótulos do eixo X
    nomes_dataframes_sem_scenario = [nome.replace(scenario, '') for nome in dataframes.keys()]

    # Define cores para cada DataFrame
    cores = plt.get_cmap('tab20', len(nomes_dataframes_sem_scenario))

    # Cria o gráfico de dispersão
    plt.figure(figsize=(12, 8))

    # Itera sobre cada DataFrame
    for i, (nome_df, df) in enumerate(dataframes.items()):
        # Itera sobre cada arquivo Java
        for nome_arquivo in nomes_arquivos:
            # Obtém os scores para o arquivo Java atual
            score = df.loc[nome_arquivo, 'score']

            # Plota o ponto com a cor correspondente ao DataFrame
            plt.scatter(i, score, color=cores(i), label=nome_df.replace(scenario,'') if nome_arquivo == nomes_arquivos[0] else None)

    # Adiciona legenda e rótulos
    plt.xticks(range(len(dataframes)), nomes_dataframes_sem_scenario, rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title(f'Dispersão de Scores para Todos os DataFrames ({scenario})')
    plt.legend(title='DataFrames', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Salva o gráfico
    plt.tight_layout()  # Ajusta o layout para evitar cortes na legenda
    plt.savefig(f'graficos/dispersao_unico_dataframes_{scenario}.png', dpi=300)
    plt.close()

def criar_boxplot_lado_a_lado(scenario, dataframes):
    # Obtém a lista de nomes dos arquivos Java
    nomes_arquivos = dataframes[list(dataframes.keys())[0]].index

    # Remove a substring do cenário dos rótulos do eixo X
    nomes_dataframes_sem_scenario = [nome.replace(scenario, '') for nome in dataframes.keys()]

    # Cria a lista de scores para cada DataFrame
    scores_por_dataframe = []
    for nome_df, df in dataframes.items():
        scores_por_dataframe.append(df['score'].values)

    # Cria o boxplot
    plt.figure(figsize=(12, 8))
    plt.boxplot(scores_por_dataframe, tick_labels=nomes_dataframes_sem_scenario)

    # Adiciona rótulos e título
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title(f'Distribuição de Scores por DataFrame ({scenario})')

    # Salva o gráfico
    plt.tight_layout()
    plt.savefig(f'graficos/boxplot_unico_dataframes_{scenario}.png', dpi=300)
    plt.close()

def distancia_frobenius(correlacoes_A, correlacoes_B):
    """Calcula a distância de Frobenius entre duas matrizes de correlação."""

    # Extrai os nomes únicos dos DataFrames
    nomes_unicos_A = sorted(list(set([nome for par in correlacoes_A.keys() for nome in par])))
    nomes_unicos_B = sorted(list(set([nome for par in correlacoes_B.keys() for nome in par])))
    nomes_unicos = sorted(list(set(nomes_unicos_A + nomes_unicos_B)))  # Combina os nomes

    # Cria matrizes para armazenar os valores de correlação
    matriz_A = pd.DataFrame(index=nomes_unicos, columns=nomes_unicos)
    matriz_B = pd.DataFrame(index=nomes_unicos, columns=nomes_unicos)

    # Preenche as matrizes com os valores de correlação
    for (nome1, nome2), valores in correlacoes_A.items():
        if valores['correlacao'] is not None:
            if valores['p_valor'] is not None and valores['p_valor'] <= 0.05:
                matriz_A.loc[nome1, nome2] = valores['correlacao']
                matriz_A.loc[nome2, nome1] = valores['correlacao']  # Correlação é simétrica
            else:
                matriz_A.loc[nome1, nome2] = 0  # Define como 0 se não for significativo
                matriz_A.loc[nome2, nome1] = 0
        else:
            matriz_A.loc[nome1, nome2] = np.nan
            matriz_A.loc[nome2, nome1] = np.nan

    for (nome1, nome2), valores in correlacoes_B.items():
        if valores['correlacao'] is not None:
            if valores['p_valor'] is not None and valores['p_valor'] <= 0.05:
                matriz_B.loc[nome1, nome2] = valores['correlacao']
                matriz_B.loc[nome2, nome1] = valores['correlacao']  # Correlação é simétrica
            else:
                matriz_B.loc[nome1, nome2] = 0  # Define como 0 se não for significativo
                matriz_B.loc[nome2, nome1] = 0
        else:
            matriz_B.loc[nome1, nome2] = np.nan
            matriz_B.loc[nome2, nome1] = np.nan

    # Converte para NumPy arrays e substitui NaN por 0
    matriz_A = matriz_A.apply(pd.to_numeric, errors='coerce').fillna(0).values
    matriz_B = matriz_B.apply(pd.to_numeric, errors='coerce').fillna(0).values

    # Calcula a distância de Frobenius
    distancia = np.linalg.norm(matriz_A - matriz_B, 'fro')
    return distancia

def correlacao_matrizes(correlacoes_A, correlacoes_B):
    """Calcula a correlação entre duas matrizes de correlação."""

    # Extrai os nomes únicos dos DataFrames
    nomes_unicos_A = sorted(list(set([nome for par in correlacoes_A.keys() for nome in par])))
    nomes_unicos_B = sorted(list(set([nome for par in correlacoes_B.keys() for nome in par])))
    nomes_unicos = sorted(list(set(nomes_unicos_A + nomes_unicos_B)))  # Combina os nomes

    # Cria matrizes para armazenar os valores de correlação
    matriz_A = pd.DataFrame(index=nomes_unicos, columns=nomes_unicos)
    matriz_B = pd.DataFrame(index=nomes_unicos, columns=nomes_unicos)

    # Preenche as matrizes com os valores de correlação
    for (nome1, nome2), valores in correlacoes_A.items():
        if valores['correlacao'] is not None:
            if valores['p_valor'] is not None and valores['p_valor'] <= 0.05:
                matriz_A.loc[nome1, nome2] = valores['correlacao']
                matriz_A.loc[nome2, nome1] = valores['correlacao']  # Correlação é simétrica
            else:
                matriz_A.loc[nome1, nome2] = 0  # Define como 0 se não for significativo
                matriz_A.loc[nome2, nome1] = 0
        else:
            matriz_A.loc[nome1, nome2] = np.nan
            matriz_A.loc[nome2, nome1] = np.nan

    for (nome1, nome2), valores in correlacoes_B.items():
        if valores['correlacao'] is not None:
            if valores['p_valor'] is not None and valores['p_valor'] <= 0.05:
                matriz_B.loc[nome1, nome2] = valores['correlacao']
                matriz_B.loc[nome2, nome1] = valores['correlacao']  # Correlação é simétrica
            else:
                matriz_B.loc[nome1, nome2] = 0  # Define como 0 se não for significativo
                matriz_B.loc[nome2, nome1] = 0
        else:
            matriz_B.loc[nome1, nome2] = np.nan
            matriz_B.loc[nome2, nome1] = np.nan

    # Converte para NumPy arrays e substitui NaN por 0
    matriz_A = matriz_A.apply(pd.to_numeric, errors='coerce').fillna(0).values.flatten()
    matriz_B = matriz_B.apply(pd.to_numeric, errors='coerce').fillna(0).values.flatten()

    # Calcula a correlação de Pearson entre os elementos das matrizes
    correlacao, _ = pearsonr(matriz_A, matriz_B)
    return correlacao

def teste_mantel(correlacoes_A, correlacoes_B, num_permutacoes=999):
    """Calcula o teste de Mantel entre duas matrizes de correlação."""

    # Extrai os nomes únicos dos DataFrames
    nomes_unicos_A = sorted(list(set([nome for par in correlacoes_A.keys() for nome in par])))
    nomes_unicos_B = sorted(list(set([nome for par in correlacoes_B.keys() for nome in par])))
    nomes_unicos = sorted(list(set(nomes_unicos_A + nomes_unicos_B)))  # Combina os nomes

    # Cria matrizes para armazenar os valores de correlação
    matriz_A = pd.DataFrame(index=nomes_unicos, columns=nomes_unicos)
    matriz_B = pd.DataFrame(index=nomes_unicos, columns=nomes_unicos)

    # Preenche as matrizes com os valores de correlação
    for (nome1, nome2), valores in correlacoes_A.items():
        if valores['correlacao'] is not None:
            if valores['p_valor'] is not None and valores['p_valor'] <= 0.05:
                matriz_A.loc[nome1, nome2] = valores['correlacao']
                matriz_A.loc[nome2, nome1] = valores['correlacao']  # Correlação é simétrica
            else:
                matriz_A.loc[nome1, nome2] = 0  # Define como 0 se não for significativo
                matriz_A.loc[nome2, nome1] = 0
        else:
            matriz_A.loc[nome1, nome2] = np.nan
            matriz_A.loc[nome2, nome1] = np.nan

    for (nome1, nome2), valores in correlacoes_B.items():
        if valores['correlacao'] is not None:
            if valores['p_valor'] is not None and valores['p_valor'] <= 0.05:
                matriz_B.loc[nome1, nome2] = valores['correlacao']
                matriz_B.loc[nome2, nome1] = valores['correlacao']  # Correlação é simétrica
            else:
                matriz_B.loc[nome1, nome2] = 0  # Define como 0 se não for significativo
                matriz_B.loc[nome2, nome1] = 0
        else:
            matriz_B.loc[nome1, nome2] = np.nan
            matriz_B.loc[nome2, nome1] = np.nan

    # Converte para NumPy arrays e substitui NaN por 0
    matriz_A = matriz_A.apply(pd.to_numeric, errors='coerce').fillna(0).values
    matriz_B = matriz_B.apply(pd.to_numeric, errors='coerce').fillna(0).values

    np.fill_diagonal(matriz_A, 1)
    np.fill_diagonal(matriz_B, 1)

    # Converte as matrizes de correlação em matrizes de distância (dissimilaridade)
    dist_A = squareform(1 - matriz_A)
    dist_B = squareform(1 - matriz_B)

    # Calcula a correlação de Mantel
    correlacao_obs = pearsonr(dist_A.flatten(), dist_B.flatten())[0]

    # Permutações
    correlacoes_perm = []
    for _ in range(num_permutacoes):
        dist_B_perm = np.random.permutation(dist_B)
        correlacao_perm = pearsonr(dist_A.flatten(), dist_B_perm.flatten())[0]
        correlacoes_perm.append(correlacao_perm)

    # Calcula o p-valor
    p_valor = (np.sum(np.array(correlacoes_perm) >= correlacao_obs) + 1) / (num_permutacoes + 1)

    return correlacao_obs, p_valor

def load_sco_data(scenario):
    dados = carregar_dados_csv(f'SCO/{scenario}.csv')
    dados['name'] = dados['name'].str.replace('/', '-') + '-score'
    dados = dados.set_index('name')
    dados['score'] = dados['score'] * 10
    return dados

def compara_matrizes_correlacao(correlacao_por_scenario):
    distancia = distancia_frobenius(correlacao_por_scenario['Original'], correlacao_por_scenario['NoComments'])
    corelacao_matriz = correlacao_matrizes(correlacao_por_scenario['Original'], correlacao_por_scenario['NoComments'])
    correlacao_obs, p_valor = teste_mantel(correlacao_por_scenario['Original'], correlacao_por_scenario['NoComments'])
    print('Semelhança entre Original e NoComments')
    print(f'Diatancia de Frobenius: {distancia}, correlação de matrizes: {corelacao_matriz} e Teste de Mantel: {correlacao_obs} p-value:{p_valor}')
    distancia = distancia_frobenius(correlacao_por_scenario['Original'], correlacao_por_scenario['BadNames'])
    corelacao_matriz = correlacao_matrizes(correlacao_por_scenario['Original'], correlacao_por_scenario['BadNames'])
    correlacao_obs, p_valor = teste_mantel(correlacao_por_scenario['Original'], correlacao_por_scenario['BadNames'])
    print('Semelhança entre Original e BadNames')
    print(f'Diatancia de Frobenius: {distancia}, correlação de matrizes: {corelacao_matriz} e Teste de Mantel: {correlacao_obs} p-value:{p_valor}')
    distancia = distancia_frobenius(correlacao_por_scenario['NoComments'], correlacao_por_scenario['BadNames'])
    corelacao_matriz = correlacao_matrizes(correlacao_por_scenario['NoComments'], correlacao_por_scenario['BadNames'])
    correlacao_obs, p_valor = teste_mantel(correlacao_por_scenario['NoComments'], correlacao_por_scenario['BadNames'])
    print('Semelhança entre NoComments e BadNames')
    print(f'Diatancia de Frobenius: {distancia}, correlação de matrizes: {corelacao_matriz} e Teste de Mantel: {correlacao_obs} p-value:{p_valor}')

def calculate_accuracy_and_roc(df):
    correct_answers = df['All']
    results = {}

    for column in df.columns:
        if column != 'All' and column != 'File':
            predicted_answers = df[column]
            accuracy = accuracy_score(correct_answers, predicted_answers)
            results[column] = {'accuracy': accuracy}

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(correct_answers, predicted_answers)
            roc_auc = auc(fpr, tpr)
            results[column]['fpr'] = fpr
            results[column]['tpr'] = tpr
            results[column]['roc_auc'] = roc_auc

            # Plot ROC curve
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {column}')
            plt.legend(loc="lower right")
            plt.savefig(f'graficos/roc-{column.replace('/', '-')}.png', dpi=300)
            plt.close()

    # Print accuracy results
    print("Accuracy Results:")
    for column, metrics in results.items():
        print(f"{column}: Accuracy = {metrics['accuracy']:.4f}")

    return results

def calculate_p_r_f1(df: pd.DataFrame, ground_truth_series: pd.Series):
    """
    Calcula Precision, Recall e F1-score para cada coluna em um DataFrame
    em comparação com uma série de ground truth.

    Args:
        df (pd.DataFrame): O DataFrame contendo as predições de cada classificador.
        ground_truth_series (pd.Series): A série contendo os valores verdadeiros (ground truth).
    """
    results = {}
    ground_truth_name = ground_truth_series.name

    # Itera sobre todas as colunas do DataFrame
    for column in df.columns:
        # Pula colunas que não são classificadores ou que são o próprio ground truth
        if column == ground_truth_name or column == 'File':
            continue

        predicted_answers = df[column]

        # Calcula as métricas. O parâmetro zero_division=0 evita warnings caso não haja
        # predições positivas, resultando em 0 para a métrica em questão.
        precision = precision_score(ground_truth_series, predicted_answers, zero_division=0)
        recall = recall_score(ground_truth_series, predicted_answers, zero_division=0)
        f1 = f1_score(ground_truth_series, predicted_answers, zero_division=0)

        results[column] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    # Imprime os resultados de forma organizada
    print(f"Métricas calculadas usando '{ground_truth_name}' como Ground Truth:\n" + "=" * 60)
    for predictor, metrics in results.items():
        print(f"{predictor:<15}: "
              f"Precision = {metrics['precision']:.4f}, "
              f"Recall = {metrics['recall']:.4f}, "
              f"F1-score = {metrics['f1_score']:.4f}")
    print("=" * 60)

    return results


def display_results_like_paper(metrics_results: dict):
    """Formata os resultados para se parecerem com a tabela do artigo."""

    # Mapeia os labels numéricos para os nomes das classes
    class_map = {-1: 'Unreadable', 0: 'Neutral', 1: 'Readable'}

    for predictor, metrics in metrics_results.items():
        print(f"--- Resultados para: {predictor} ---")

        # Cria um DataFrame para fácil visualização
        display_df = pd.DataFrame(columns=['Precision', 'Recall', 'F1-score'])

        # Adiciona as linhas por classe
        for label, class_metrics in metrics['per_class'].items():
            class_name = class_map.get(label, f"Class {label}")
            display_df.loc[class_name] = class_metrics

        # Adiciona a linha de média
        display_df.loc['Average (Macro)'] = metrics['average']

        # Imprime o DataFrame formatado
        print(display_df.to_string(float_format="%.2f"))
        print("\n")

def calculate_multiclass_metrics(df: pd.DataFrame, ground_truth_series: pd.Series):
    results = {}
    ground_truth_name = ground_truth_series.name
    class_labels = sorted(ground_truth_series.unique())

    for column in df.columns:
        if column == ground_truth_name:
            continue

        predicted_answers = df[column]

        p_per_class = precision_score(ground_truth_series, predicted_answers, labels=class_labels, average=None, zero_division=0)
        r_per_class = recall_score(ground_truth_series, predicted_answers, labels=class_labels, average=None, zero_division=0)
        f1_per_class = f1_score(ground_truth_series, predicted_answers, labels=class_labels, average=None, zero_division=0)

        macro_p = precision_score(ground_truth_series, predicted_answers, labels=class_labels, average='macro', zero_division=0)
        macro_r = recall_score(ground_truth_series, predicted_answers, labels=class_labels, average='macro', zero_division=0)
        macro_f1 = f1_score(ground_truth_series, predicted_answers, labels=class_labels, average='macro', zero_division=0)

        results[column] = {'per_class': {}, 'average': {}}
        for i, label in enumerate(class_labels):
            results[column]['per_class'][label] = {
                'Precision': p_per_class[i], 'Recall': r_per_class[i], 'F1-score': f1_per_class[i],
            }
        results[column]['average'] = {
            'Precision': macro_p, 'Recall': macro_r, 'F1-score': macro_f1,
        }
    return results

def plot_java_file_scores(df, scenario):
    plt.figure(figsize=(12, 8))  # Ajuste o tamanho da figura conforme necessário

    for index, row in df.iterrows():
        plt.plot(row.values, marker='o', label=index)

    plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha='right')  # Rotaciona os rótulos do eixo x
    plt.xlabel('Colunas (Modelos)')
    plt.ylabel('Pontuação')
    plt.ylim(0, 10)
    plt.title(f'Pontuações dos Arquivos Java por Modelo - Cenário {scenario}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Coloca a legenda fora do gráfico
    plt.grid(True)
    plt.tight_layout()  # Ajusta o layout para evitar cortes nos rótulos
    plt.savefig(f'graficos/rank-{scenario}.png', dpi=300)
    plt.close()

def combine_lists_to_dataframe(data_dict):
    combined_data = {}
    for name, df in data_dict.items():
        for index, row in df.iterrows():
            file_name = re.sub(r'-score$', '', index)  # Remove '-score' from index
            file_name = re.sub(r'^BadNames-AnoBissexto', 'E4-AnoBissexto', file_name)
            file_name = re.sub(r'^BadNames-CalculaAreaCirculo', 'E4-CalculaAreaCirculo', file_name)
            file_name = re.sub(r'^BadNames-CalculaAreaTrianguloIsoceles', 'E4-CalculaAreaTrianguloIsoceles', file_name)
            file_name = re.sub(r'^BadNames-CalculaDVBase10', 'E4-CalculaDVBase10', file_name)
            file_name = re.sub(r'^BadNames-GetMedia', 'E3-GetMedia', file_name)
            file_name = re.sub(r'^BadNames-MonthSubtrai', 'E3-MonthSubtrai', file_name)
            file_name = re.sub(r'^BadNames-MonthPlus', 'E3-MonthPlus', file_name)
            file_name = re.sub(r'^BadNames-MonthPrimeiroMesDoTrimestre', 'E3-MonthPrimeiroMesDoTrimestre', file_name)
            file_name = re.sub(r'^BadNames-Combine', 'E3-Combine', file_name)
            file_name = re.sub(r'^BadNames-CarregarFilhos', 'E3-CarregarFilhos', file_name)

            file_name = re.sub(r'^NoComments-AnoBissexto', 'E2-AnoBissexto', file_name)
            file_name = re.sub(r'^NoComments-CalculaAreaCirculo', 'E2-CalculaAreaCirculo', file_name)
            file_name = re.sub(r'^NoComments-CalculaAreaTrianguloIsoceles', 'E2-CalculaAreaTrianguloIsoceles', file_name)
            file_name = re.sub(r'^NoComments-GetMedia', 'E2-GetMedia', file_name)
            file_name = re.sub(r'^NoComments-MonthSubtrai', 'E2-MonthSubtrai', file_name)
            file_name = re.sub(r'^NoComments-MonthPlus', 'E2-MonthPlus', file_name)
            file_name = re.sub(r'^NoComments-MonthPrimeiroMesDoTrimestre', 'E2-MonthPrimeiroMesDoTrimestre', file_name)
            file_name = re.sub(r'^NoComments-Combine', 'E2-Combine', file_name)
            file_name = re.sub(r'^NoComments-CarregarFilhos', 'E2-CarregarFilhos', file_name)
            file_name = re.sub(r'^NoComments-CalculaDVBase10', 'E2-CalculaDVBase10', file_name)

            file_name = re.sub(r'^Original-AnoBissexto', 'E1-AnoBissexto', file_name)
            file_name = re.sub(r'^Original-CalculaAreaCirculo', 'E1-CalculaAreaCirculo', file_name)
            file_name = re.sub(r'^Original-CalculaAreaTrianguloIsoceles', 'E1-CalculaAreaTrianguloIsoceles', file_name)
            file_name = re.sub(r'^Original-GetMedia', 'E1-GetMedia', file_name)
            file_name = re.sub(r'^Original-MonthSubtrai', 'E1-MonthSubtrai', file_name)
            file_name = re.sub(r'^Original-MonthPlus', 'E1-MonthPlus', file_name)
            file_name = re.sub(r'^Original-MonthPrimeiroMesDoTrimestre', 'E1-MonthPrimeiroMesDoTrimestre', file_name)
            file_name = re.sub(r'^Original-Combine', 'E1-Combine', file_name)
            file_name = re.sub(r'^Original-CarregarFilhos', 'E1-CarregarFilhos', file_name)
            file_name = re.sub(r'^Original-CalculaDVBase10', 'E1-CalculaDVBase10', file_name)
            if file_name not in combined_data:
                combined_data[file_name] = {}
            combined_data[file_name][name] = row['score']

    return pd.DataFrame(combined_data).T


def combine_lists_to_dataframe_std(data_dict):
    """
    Combina um dicionário de DataFrames de desvio padrão em um único DataFrame.

    Assume que cada DataFrame em data_dict tem uma coluna chamada 'std'.
    Se a coluna 'std' não existir (ex: para 'SCO'), preenche com NaN.
    """
    combined_data = {}
    for name, df in data_dict.items():
        for index, row in df.iterrows():
            file_name = re.sub(r'-score$', '', index)  # Remove '-score' from index
            file_name = re.sub(r'^BadNames-AnoBissexto', 'E4-AnoBissexto', file_name)
            file_name = re.sub(r'^BadNames-CalculaAreaCirculo', 'E4-CalculaAreaCirculo', file_name)
            file_name = re.sub(r'^BadNames-CalculaAreaTrianguloIsoceles', 'E4-CalculaAreaTrianguloIsoceles', file_name)
            file_name = re.sub(r'^BadNames-CalculaDVBase10', 'E4-CalculaDVBase10', file_name)
            file_name = re.sub(r'^BadNames-GetMedia', 'E3-GetMedia', file_name)
            file_name = re.sub(r'^BadNames-MonthSubtrai', 'E3-MonthSubtrai', file_name)
            file_name = re.sub(r'^BadNames-MonthPlus', 'E3-MonthPlus', file_name)
            file_name = re.sub(r'^BadNames-MonthPrimeiroMesDoTrimestre', 'E3-MonthPrimeiroMesDoTrimestre', file_name)
            file_name = re.sub(r'^BadNames-Combine', 'E3-Combine', file_name)
            file_name = re.sub(r'^BadNames-CarregarFilhos', 'E3-CarregarFilhos', file_name)

            file_name = re.sub(r'^NoComments-AnoBissexto', 'E2-AnoBissexto', file_name)
            file_name = re.sub(r'^NoComments-CalculaAreaCirculo', 'E2-CalculaAreaCirculo', file_name)
            file_name = re.sub(r'^NoComments-CalculaAreaTrianguloIsoceles', 'E2-CalculaAreaTrianguloIsoceles', file_name)
            file_name = re.sub(r'^NoComments-GetMedia', 'E2-GetMedia', file_name)
            file_name = re.sub(r'^NoComments-MonthSubtrai', 'E2-MonthSubtrai', file_name)
            file_name = re.sub(r'^NoComments-MonthPlus', 'E2-MonthPlus', file_name)
            file_name = re.sub(r'^NoComments-MonthPrimeiroMesDoTrimestre', 'E2-MonthPrimeiroMesDoTrimestre', file_name)
            file_name = re.sub(r'^NoComments-Combine', 'E2-Combine', file_name)
            file_name = re.sub(r'^NoComments-CarregarFilhos', 'E2-CarregarFilhos', file_name)
            file_name = re.sub(r'^NoComments-CalculaDVBase10', 'E2-CalculaDVBase10', file_name)

            file_name = re.sub(r'^Original-AnoBissexto', 'E1-AnoBissexto', file_name)
            file_name = re.sub(r'^Original-CalculaAreaCirculo', 'E1-CalculaAreaCirculo', file_name)
            file_name = re.sub(r'^Original-CalculaAreaTrianguloIsoceles', 'E1-CalculaAreaTrianguloIsoceles', file_name)
            file_name = re.sub(r'^Original-GetMedia', 'E1-GetMedia', file_name)
            file_name = re.sub(r'^Original-MonthSubtrai', 'E1-MonthSubtrai', file_name)
            file_name = re.sub(r'^Original-MonthPlus', 'E1-MonthPlus', file_name)
            file_name = re.sub(r'^Original-MonthPrimeiroMesDoTrimestre', 'E1-MonthPrimeiroMesDoTrimestre', file_name)
            file_name = re.sub(r'^Original-Combine', 'E1-Combine', file_name)
            file_name = re.sub(r'^Original-CarregarFilhos', 'E1-CarregarFilhos', file_name)
            file_name = re.sub(r'^Original-CalculaDVBase10', 'E1-CalculaDVBase10', file_name)

            if file_name not in combined_data:
                combined_data[file_name] = {}

            # --- Esta é a única linha alterada ---
            # Acessa 'std' em vez de 'score' e trata casos onde 'std' não existe (ex: SCO)
            if 'std' in row:
                combined_data[file_name][name] = row['std']
            # else:
            #     combined_data[file_name][name] = np.nan

    return pd.DataFrame(combined_data).T

def gerar_latex_tabelas_resumo(df: pd.DataFrame):
    """
    Gera e imprime o código LaTeX para três tabelas de resumo a partir de um DataFrame.

    As tabelas sumarizam as colunas 'experience', 'role' e 'language'.

    Args:
        df (pd.DataFrame): O DataFrame de entrada contendo os dados dos respondentes.
                           Deve incluir as colunas 'experience', 'role' e 'language'.
    """
    # --- 1. Preparação e Limpeza dos Dados ---

    # Copia o dataframe para evitar modificar o original (boa prática)
    df_proc = df.copy()

    # Normaliza o cargo 'Mid' para 'Pleno' para corresponder à tabela LaTeX
    if 'role' in df_proc.columns:
        df_proc['role'] = df_proc['role'].replace('Mid', 'Pleno')

    total_records = len(df_proc)

    # Define a ordem exata das categorias para cada tabela, conforme o template LaTeX
    ordem_experiencia = [
        'Menos de 2 anos',
        'Entre 2 e 5 anos',
        'Entre 5 e 10 anos',
        'Entre 10 e 20 anos',
        'Mais de 20 anos'
    ]

    ordem_cargo = [
        'Junior',
        'Pleno',
        'Senior',
        'Staff',
        'Principal'
    ]

    ordem_linguagem = [
        'Java',
        'Javascript/Typescript',
        'Python',
        'C/C++',
        'Cobol',
        'Outra'
    ]

    # --- 2. Função Auxiliar para Gerar as Linhas da Tabela ---

    def criar_linhas_tabela(column_name: str, category_order: list) -> str:
        """Gera as linhas de dados para uma tabela LaTeX."""
        linhas = []

        # Calcula a contagem de valores para a coluna especificada
        counts = df_proc[column_name].value_counts()

        for category in category_order:
            count = counts.get(category, 0)
            percentage = (count / total_records) * 100 if total_records > 0 else 0
            # Formata a linha da tabela LaTeX. Note o escape duplo '\\' para a contrabarra
            # e '%' para o sinal de porcentagem.
            linhas.append(f"        {category} & {count} & {percentage:.0f}\\%\\\\")

        return "\n".join(linhas)

    # --- 3. Geração das Linhas para Cada Tabela ---

    linhas_tabela_experiencia = criar_linhas_tabela('experience', ordem_experiencia)
    linhas_tabela_cargo = criar_linhas_tabela('role', ordem_cargo)
    linhas_tabela_linguagem = criar_linhas_tabela('language', ordem_linguagem)

    # --- 4. Montagem do Código LaTeX Final ---

    latex_output = f"""
\\begin{{table}}[ht]
\\begin{{minipage}}{{.55\\linewidth}}
    \\footnotesize
        \\begin{{tabular}}{{lcc}}
        \\toprule
        \\textbf{{Resposta}} & \\textbf{{Quantidade}} & \\%\\\\
        \\midrule
{linhas_tabela_experiencia}
        \\bottomrule
        \\end{{tabular}}
        \\caption{{Experiência}}
    \\label{{tab:respondentesExperiencia}}
\\end{{minipage}}
\\begin{{minipage}}{{.44\\linewidth}}
    \\footnotesize
        \\begin{{tabular}}{{lcc}}
        \\toprule
        \\textbf{{Resposta}} & \\textbf{{Quantidade}} & \\%\\\\
        \\midrule
{linhas_tabela_cargo}
        \\bottomrule
        \\end{{tabular}}
        \\caption{{Cargo}}
    \\label{{tab:respondentesCargo}}
\\end{{minipage}}
\\end{{table}}

\\begin{{table}}[ht]
    \\footnotesize
        \\begin{{tabular}}{{lcc}}
        \\toprule
        \\textbf{{Resposta}} & \\textbf{{Quantidade}} & \\%\\\\
        \\midrule
{linhas_tabela_linguagem}
        \\bottomrule
        \\end{{tabular}}
        \\caption{{Linguagem Principal}}
    \\label{{tab:respondentesLinguagem}}
\\end{{table}}
"""

    # Imprime o resultado final
    print(latex_output)


def gerar_latex_experiencia_por_cargo(df: pd.DataFrame):
    """
    Gera e imprime o código LaTeX para uma tabela de contingência mostrando
    a distribuição do tempo de experiência por cargo.

    Os valores na tabela são percentuais calculados por coluna (por cargo).

    Args:
        df (pd.DataFrame): O DataFrame de entrada. Deve conter as colunas
                           'experience' e 'role'.
    """
    # --- 1. Preparação e Limpeza dos Dados ---

    # Copia o dataframe para evitar modificar o original
    df_proc = df.copy()

    # Normaliza o cargo 'Mid' para 'Pleno'
    if 'role' in df_proc.columns:
        df_proc['role'] = df_proc['role'].replace('Mid', 'Pleno')

    # Define a ordem exata das categorias para as linhas e colunas
    ordem_experiencia = [
        'Menos de 2 anos', 'Entre 2 e 5 anos', 'Entre 5 e 10 anos',
        'Entre 10 e 20 anos', 'Mais de 20 anos'
    ]
    ordem_cargo = ['Junior', 'Pleno', 'Senior', 'Staff', 'Principal']

    # --- 2. Criação da Tabela de Contingência (Crosstab) ---

    # Cria a tabela de contingência com contagens brutas
    # Usa reindex para garantir que todas as categorias e ordens estejam corretas
    try:
        tabela_contas = pd.crosstab(df_proc['experience'], df_proc['role'])
        tabela_contas = tabela_contas.reindex(index=ordem_experiencia, columns=ordem_cargo, fill_value=0)
    except KeyError:
        print("Erro: As colunas 'experience' ou 'role' não foram encontradas no DataFrame.")
        return

    # Calcula os percentuais por coluna (cada coluna soma 100%)
    # Divide-se cada valor pela soma de sua respectiva coluna
    soma_colunas = tabela_contas.sum(axis=0)
    # Evita divisão por zero caso uma coluna inteira seja 0
    tabela_percentual = tabela_contas.div(soma_colunas, axis=1).fillna(0) * 100

    # --- 3. Geração do Código LaTeX ---

    # Cria o cabeçalho da tabela dinamicamente
    header_cols = ' & '.join([f'\\textbf{{{col}}}' for col in tabela_percentual.columns])
    latex_header = f"\\textbf{{Resposta}} & {header_cols}\\\\"

    # Cria as linhas de dados da tabela
    linhas_tabela = []
    for experience_level, row in tabela_percentual.iterrows():
        # Formata cada percentual na linha
        formatted_values = [f"{val:.0f}\\%" for val in row.values]
        # Une a categoria de experiência com seus valores percentuais
        row_string = f"        {experience_level} & {' & '.join(formatted_values)}\\\\"
        linhas_tabela.append(row_string)

    # Junta todas as linhas de dados em um único bloco de texto
    corpo_tabela = "\n".join(linhas_tabela)

    # Monta o template final do LaTeX com os dados gerados
    latex_output = f"""
\\begin{{table}}[ht]
    \\footnotesize
    \\centering
        \\begin{{tabular}}{{lccccc}}
        \\toprule
        {latex_header}
        \\midrule
{corpo_tabela}
        \\bottomrule
        \\end{{tabular}}
        \\caption{{Tempo programando por cargo}}
    \\label{{tab:respondentesTempoPorCargo}}
\\end{{table}}
"""

    # Imprime o resultado final
    print(latex_output)


def plotar_distribuicao_cargo_experiencia(df: pd.DataFrame, caminho_salvar: str = 'graficos/distribuicaoCargosExperiencia.png'):
    """
    Gera e salva um gráfico de barras empilhadas mostrando a distribuição
    percentual do tempo de experiência para cada cargo.

    Args:
        df (pd.DataFrame): O DataFrame de entrada. Deve conter as colunas
                           'experience' e 'role'.
        caminho_salvar (str): O caminho completo, incluindo o nome do arquivo,
                              onde o gráfico será salvo.
    """
    # --- 1. Preparação e Limpeza dos Dados ---

    print("Iniciando a geração do gráfico...")
    df_proc = df.copy()

    # Normaliza o cargo 'Mid' para 'Pleno'
    if 'role' in df_proc.columns:
        df_proc['role'] = df_proc['role'].replace('Mid', 'Pleno')

    # Define a ordem exata das categorias
    ordem_experiencia = [
        'Menos de 2 anos', 'Entre 2 e 5 anos', 'Entre 5 e 10 anos',
        'Entre 10 e 20 anos', 'Mais de 20 anos'
    ]
    ordem_cargo = ['Junior', 'Pleno', 'Senior', 'Staff', 'Principal']

    # --- 2. Criação da Tabela de Contingência (Crosstab) com Percentuais ---
    try:
        tabela_percentual = pd.crosstab(
            df_proc['experience'], df_proc['role'], normalize='columns'
        ) * 100
        # Reindexa para garantir a ordem correta no gráfico e na legenda
        tabela_percentual = tabela_percentual.reindex(
            index=ordem_experiencia, columns=ordem_cargo, fill_value=0
        )
    except KeyError:
        print("Erro: As colunas 'experience' ou 'role' não foram encontradas no DataFrame.")
        return

    # --- 3. Geração do Gráfico ---

    # Define um tema estético para o gráfico
    sns.set_theme(style="whitegrid")

    # Cria a figura e os eixos do Matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plota os dados transpostos como um gráfico de barras empilhadas
    # Usamos .T para ter os 'cargos' no eixo X
    tabela_percentual.T.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        colormap='viridis'  # Escolha de um mapa de cores visualmente agradável
    )

    # --- 4. Customização do Gráfico ---

    # Títulos e rótulos
    ax.set_title('Distribuição Percentual de Experiência por Cargo', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Cargo', fontsize=12)
    ax.set_ylabel('Distribuição Percentual', fontsize=12)

    # Formata o eixo Y para mostrar o símbolo de porcentagem
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # Rotaciona os rótulos do eixo X para melhor visualização
    plt.xticks(rotation=0, ha='center')

    # Customiza a legenda
    ax.legend(
        title='Nível de Experiência',
        bbox_to_anchor=(1.02, 1),  # Posiciona a legenda fora da área do gráfico
        loc='upper left'
    )

    # Ajusta o layout para evitar sobreposições
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajusta para dar espaço à legenda externa

    # --- 5. Salvar o Arquivo ---

    try:
        # Extrai o diretório do caminho fornecido
        output_dir = os.path.dirname(caminho_salvar)

        # Cria o diretório se ele não existir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Diretório '{output_dir}' criado.")

        # Salva a figura
        plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')

        print(f"Gráfico salvo com sucesso em '{caminho_salvar}'")

    except Exception as e:
        print(f"Ocorreu um erro ao salvar o arquivo: {e}")
    finally:
        # Fecha a figura para liberar memória
        plt.close(fig)


def calculate_mcdonalds_omega_overall(df: pd.DataFrame):
    """
    Calcula o Omega de McDonald para um conjunto de dados, usando um método
    robusto ('minres') para evitar erros de convergência.
    """
    df_wide = df.transpose()
    is_duplicate = df_wide.T.duplicated()
    df_unique_runs = df_wide.loc[:, ~is_duplicate]

    num_unique_runs = df_unique_runs.shape[1]

    if is_duplicate.any():
        num_removed = is_duplicate.sum()
        # print(f"AVISO: {num_removed} execução(ões) redundante(s) foram removidas.")

    if num_unique_runs < 2:
        # print("INFO: Todas as execuções foram idênticas. A confiabilidade é perfeita.")
        return 1.0

    if num_unique_runs < 3:
        # print("Apenas duas execuções únicas foram encontradas. Omega não pode ser calculado, mas a variabilidade é baixa.")
        return 1.0

    # --- ALTERAÇÃO FINAL PARA ROBUSTEZ ---
    # Trocando o método 'ml' por 'minres' para evitar falhas de convergência numérica.
    fa = FactorAnalyzer(n_factors=1, rotation=None, method='minres')
    fa.fit(df_unique_runs)

    loadings = fa.loadings_
    error_variances = fa.get_uniquenesses()

    sum_of_loadings_sq = np.sum(loadings) ** 2
    sum_of_error_variances = np.sum(error_variances)

    # Adicionando uma verificação para o denominador para evitar divisão por zero
    denominator = sum_of_loadings_sq + sum_of_error_variances
    if denominator == 0:
        return 1.0 if sum_of_error_variances == 0 else 0.0

    omega = sum_of_loadings_sq / denominator
    return omega


def plot_binary_metrics(metrics_results: dict,
                        sort_by: str = 'F1-score',
                        ascending: bool = False):
    """
    Gera um gráfico de barras agrupadas para as métricas de classificação binária,
    com opção de ordenação.

    Args:
        metrics_results (dict): O dicionário retornado por 'calculate_p_r_f1'.
        save_path (str): Caminho para salvar o gráfico.
        sort_by (str, optional): A métrica ('Precision', 'Recall', 'F1-score')
                                 pela qual ordenar o gráfico.
                                 Default é 'F1-score'. Use None para não ordenar.
        ascending (bool, optional): Ordem da classificação (ascendente).
                                    Default é False (descendente).
    """
    save_path = f'graficos/binary_classification_metrics_{sort_by}.png'
    print(f"Gerando gráfico de métricas binárias em '{save_path}'...")

    # 1. Converter o dicionário de métricas para um DataFrame
    df = pd.DataFrame.from_dict(metrics_results, orient='index')

    # Padronizar nomes das colunas para o gráfico
    df.rename(columns={
        'f1_score': 'F1-score',
        'precision': 'Precision',
        'recall': 'Recall'
    }, inplace=True)

    # 2. Definir a ordem de classificação (NOVA LÓGICA)
    group_order = None
    if sort_by:
        if sort_by in df.columns:
            df_sorted = df.sort_values(by=sort_by, ascending=ascending)
            group_order = df_sorted.index.tolist()
        else:
            print(f"Aviso: A coluna de ordenação '{sort_by}' não foi encontrada. "
                  f"Valores válidos: {list(df.columns)}")

    # 3. Reformatar para o formato "longo" (ideal para Seaborn)
    df_long = df.reset_index().rename(columns={'index': 'Grupo'})
    df_long = df_long.melt('Grupo', var_name='Métrica', value_name='Score')

    # 4. Criar o gráfico
    g = sns.catplot(
        data=df_long,
        kind='bar',
        x='Grupo',
        y='Score',
        hue='Métrica',
        height=7,
        aspect=2,
        palette='viridis',
        order=group_order  # <-- Passa a ordem para o gráfico
    )

    # 5. Ajustes e anotações
    g.ax.set_title('Desempenho da Classificação Binária (Precision, Recall, F1-score)', fontsize=16)
    g.set_ylabels('Pontuação (Score)', fontsize=12)
    g.set_xlabels('Grupo / Modelo', fontsize=12)
    g.set_xticklabels(rotation=45, ha='right')
    g.ax.set_ylim(0, 1.1)
    g.despine(left=True)

    # Adicionar os valores nas barras
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontsize=8,
                        weight='bold')

    # 6. Salvar o gráfico
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Gráfico de métricas binárias salvo.")

def plot_ternary_metrics(metrics_results: dict,
                         sort_by: str = 'F1-score',
                         ascending: bool = False):
    """
    Gera um gráfico de barras agrupadas para as métricas de classificação ternária
    (focando na Média Macro), com opção de ordenação.

    Args:
        metrics_results (dict): O dicionário retornado por 'calculate_multiclass_metrics'.
        save_path (str): Caminho para salvar o gráfico.
        sort_by (str, optional): A métrica ('Precision', 'Recall', 'F1-score')
                                 pela qual ordenar o gráfico.
                                 Default é 'F1-score'. Use None para não ordenar.
        ascending (bool, optional): Ordem da classificação (ascendente).
                                    Default é False (descendente).
    """
    save_path = f'graficos/ternary_classification_metrics_{sort_by}.png'
    print(f"Gerando gráfico de métricas ternárias em '{save_path}'...")

    # 1. Extrair apenas as métricas "average" (Média Macro) do dicionário
    avg_metrics = {model: data['average'] for model, data in metrics_results.items()}

    # 2. Converter o dicionário extraído para um DataFrame
    df = pd.DataFrame.from_dict(avg_metrics, orient='index')
    # Os nomes das colunas ('Precision', 'Recall', 'F1-score') já estão corretos

    # 3. Definir a ordem de classificação (NOVA LÓGICA)
    group_order = None
    if sort_by:
        if sort_by in df.columns:
            df_sorted = df.sort_values(by=sort_by, ascending=ascending)
            group_order = df_sorted.index.tolist()
        else:
            print(f"Aviso: A coluna de ordenação '{sort_by}' não foi encontrada. "
                  f"Valores válidos: {list(df.columns)}")

    # 4. Reformatar para o formato "longo"
    df_long = df.reset_index().rename(columns={'index': 'Grupo'})
    df_long = df_long.melt('Grupo', var_name='Métrica', value_name='Score')

    # 5. Criar o gráfico
    g = sns.catplot(
        data=df_long,
        kind='bar',
        x='Grupo',
        y='Score',
        hue='Métrica',
        height=7,
        aspect=2,
        palette='plasma',
        order=group_order  # <-- Passa a ordem para o gráfico
    )

    # 6. Ajustes e anotações
    g.ax.set_title('Desempenho da Classificação Ternária (Média Macro)', fontsize=16)
    g.set_ylabels('Pontuação (Score)', fontsize=12)
    g.set_xlabels('Grupo / Modelo', fontsize=12)
    g.set_xticklabels(rotation=45, ha='right')
    g.ax.set_ylim(0, 1.1)
    g.despine(left=True)

    # Adicionar os valores nas barras
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontsize=8,
                        weight='bold')

    # 7. Salvar o gráfico
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Gráfico de métricas ternárias salvo.")

def interpret_reliability(omega_value):
    """Interpreta o valor de confiabilidade."""
    if pd.isna(omega_value):
        return "Incalculável"
    if omega_value > 0.9:
        return 'Excelente'
    elif 0.8 < omega_value <= 0.9:
        return 'Bom'
    elif 0.7 < omega_value <= 0.8:
        return 'Aceitável'
    else:
        return 'Questionável ou Pobre'


def classify_ternary_around_mean(df_scores: pd.DataFrame,
                                 global_mean: float,
                                 global_std: float,
                                 neutral_band_std_width: float = 0.5) -> pd.DataFrame:
    """
    Classifica scores de um DataFrame em três categorias usando um ponto central (média global)
    e uma faixa neutra definida pelo desvio padrão global.

    A faixa neutra é definida como: [média - (largura * std), média + (largura * std)].
    Por padrão, a largura é 0.5, significando que a faixa vai de -0.25*std a +0.25*std
    em torno da média, cobrindo 50% de uma distribuição normal.

    Args:
        df_scores (pd.DataFrame): DataFrame contendo os scores a serem classificados.
        global_mean (float): O ponto de corte central (sua LIMITE_LEGIVEL).
        global_std (float): O desvio padrão da população de scores que gerou a média.
        neutral_band_std_width (float): A largura total da faixa neutra em unidades de
                                        desvio padrão. Default = 0.5.

    Returns:
        pd.DataFrame: Um novo DataFrame com os scores classificados em 1 (Legível),
                      0 (Neutro), e -1 (Ilegível).
    """
    # 1. Calcular os limites da faixa neutra
    half_width = (neutral_band_std_width / 2) * global_std
    lower_bound = 3.5 #global_mean - half_width
    upper_bound = 7 #global_mean + half_width

    print("=" * 60)
    print(f"Usando Média Global como pivô para classificação ternária:")
    print(f"  - Média (Ponto Central): {global_mean:.4f}")
    print(f"  - Desvio Padrão Global: {global_std:.4f}")
    print(f"  - Limite Inferior (Ilegível <=): {lower_bound:.4f}")
    print(f"  - Limite Superior (Legível >=): {upper_bound:.4f}")
    print("=" * 60 + "\n")

    # 2. Definir a função de classificação baseada nestes limites fixos
    def _classify_score(score):
        if score >= upper_bound:
            return 1  # Legível
        elif score <= lower_bound:
            return -1  # Ilegível
        else:
            return 0  # Neutro

    # 3. Aplicar a função a todo o DataFrame de scores
    df_classified = df_scores.map(_classify_score)

    return df_classified.astype(int)

def statisticas_por_cluster(attribute):
    global role
    for role in roles:
        role_medias = []
        role_stds = []
        for group in df_group:
            # print(group)
            role_group = group[group[attribute] == role].drop(['Id', 'experience', 'language', 'role'], axis=1)
            if not role_group.empty:
                role_medias.append(role_group.mean())
                role_stds.append(role_group.std())
        if role_medias:
            medias[role] = pd.concat(role_medias, axis=1).mean(axis=1)
            std[role] = pd.concat(role_stds, axis=1).mean(axis=1)
        else:
            medias[role] = pd.Series()
            std[role] = pd.Series()

def calculate_krippendorff_alpha_by_group(df):
    """
    Calcula o Alpha de Krippendorff para grupos predefinidos de colunas
    com base em seus prefixos de nome (Original-, BadNames-, NoComments-).

    Argumentos:
        df (pd.DataFrame): O dataframe de entrada com avaliadores como linhas
                           e itens (pontuações) como colunas.

    Retorna:
        dict: Um dicionário com valores alfa para cada grupo e um alfa geral.
    """

    # Identificar grupos de colunas
    original_cols = [col for col in df.columns if col.startswith('Original-')]
    badnames_cols = [col for col in df.columns if col.startswith('BadNames-')]
    nocomments_cols = [col for col in df.columns if col.startswith('NoComments-')]

    all_score_cols = original_cols + badnames_cols + nocomments_cols

    # print("--- Calculando Alphas ---")
    # print(f"Encontradas {len(original_cols)} colunas 'Original': {original_cols}")
    # print(f"Encontradas {len(badnames_cols)} colunas 'BadNames': {badnames_cols}")
    # print(f"Encontradas {len(nocomments_cols)} colunas 'NoComments': {nocomments_cols}")

    results = {}

    # Função auxiliar para calcular o alfa com segurança
    def compute_alpha(data, group_name):
        # Precisa de pelo menos 2 itens (colunas) para o cálculo
        if data.empty or data.shape[1] < 2:
            print(f"Pulando '{group_name}': são necessários pelo menos 2 itens (colunas) para o cálculo.")
            return None

        # Verificar se todos os dados são numéricos, coagir se não forem
        numeric_data = data.apply(pd.to_numeric, errors='coerce')
        if numeric_data.isnull().values.any():
            print(f"Aviso: Dados não numéricos ou ausentes encontrados em '{group_name}'. Serão tratados como ausentes.")

        # A função krippendorff.alpha espera (avaliadores, itens).
        # Nosso dataframe já está como (41 avaliadores, N itens).
        # Usamos .values para passar o array numpy subjacente.
        # `level_of_measurement='interval'` é apropriado para pontuações de 0-10.
        # A biblioteca lida com valores ausentes (NaN) por padrão.
        try:
            return krippendorff.alpha(numeric_data.values, level_of_measurement='interval')
        except ValueError as e: return 'Não calculado'

    # Calcular para cada grupo
    if original_cols:
        results['Original_Items_Alpha'] = compute_alpha(df[original_cols], 'Original')

    if badnames_cols:
        results['BadNames_Items_Alpha'] = compute_alpha(df[badnames_cols], 'BadNames')

    if nocomments_cols:
        results['NoComments_Items_Alpha'] = compute_alpha(df[nocomments_cols], 'NoComments')

    # Calcular o alfa geral para todos os 10 itens
    if len(all_score_cols) > 1:
        results['Overall_Alpha'] = compute_alpha(df[all_score_cols], 'Overall')

    return results


def calculate_xrr_metrics(df_human_ratings: pd.DataFrame,
                          df_llm_ratings_runs: pd.DataFrame,
                          llm_irr: float,
                          human_irr: float = None) -> dict:
    """
    Calcula as métricas do framework Cross-replication Reliability (xRR) entre
    um pool de avaliadores humanos (com dados ausentes) e um pool de LLM (denso).

    Baseado em: Wong, Paritosh, & Aroyo (2021). "Cross-replication Reliability".
    Implementa κₓ (kappa_x) com dados ausentes (Eq. 9, 10) e κₓ normalizado (Eq. 13).

    Args:
        df_human_ratings (pd.DataFrame): DataFrame (N_humanos x N_itens) com as notas
                                         dos humanos. Deve conter 'Id' ou ser
                                         apenas colunas de score. Ex: df_notas
        df_llm_ratings_runs (pd.DataFrame): DataFrame (N_runs_llm x N_itens) com as
                                            notas do LLM. Ex: df_llm[llm]
        llm_irr (float): A confiabilidade interna pré-calculada do pool de LLMs
                         (ex: Omega de McDonald).
        human_irr (float, optional): A confiabilidade interna pré-calculada do
                                     pool humano (ex: Alpha de Krippendorff).
                                     Se None, será calculada.

    Returns:
        dict: Um dicionário contendo 'kappa_x', 'normalized_kappa_x',
              'irr_human', 'irr_llm', 'd_o' (desacordo observado),
              e 'd_e' (desacordo esperado).
    """

    # --- 1. Definir Função de Desacordo (Squared Distance) ---
    D = lambda x, y: (x - y) ** 2

    # --- 2. Preparar Pools e Calcular IRRs ---

    # Pool Humano (X)
    human_pool_df = df_human_ratings.drop('Id', axis=1, errors='ignore')  # (N_humanos x 30 itens)
    human_values = human_pool_df.values

    # Pool LLM (Y)
    llm_pool_df = df_llm_ratings_runs  # (10 runs x 30 itens)
    llm_values = llm_pool_df.values.T  # (30 itens x 10 runs)

    if human_irr is None:
        human_irr = krippendorff.alpha(human_values, level_of_measurement='interval')

    # --- 3. Calcular d_e (Desacordo Esperado - Eq. 10) ---
    all_human_ratings = human_values.flatten()
    all_human_ratings = all_human_ratings[~np.isnan(all_human_ratings)]

    all_llm_ratings = llm_values.flatten()  # Já é denso

    R_total = len(all_human_ratings)  # R total (Eq. 10)
    S_total = len(all_llm_ratings)  # S total (Eq. 11)

    if R_total == 0 or S_total == 0:
        return {'kappa_x': np.nan, 'normalized_kappa_x': np.nan,
                'irr_human': human_irr, 'irr_llm': llm_irr,
                'd_o': np.nan, 'd_e': np.nan}

    mean_H = np.mean(all_human_ratings)
    mean_L = np.mean(all_llm_ratings)
    mean_sq_H = np.mean(np.square(all_human_ratings))
    mean_sq_L = np.mean(np.square(all_llm_ratings))

    d_e = mean_sq_H + mean_sq_L - (2 * mean_H * mean_L)

    # --- 4. Calcular d_o (Desacordo Observado - Eq. 9) ---
    total_observed_disagreement = 0

    items = llm_pool_df.columns.tolist()  # <-- CORREÇÃO

    for item_name in items:
        if item_name not in human_pool_df.columns:
            continue

        human_ratings_item = human_pool_df[item_name].dropna().values

        llm_ratings_item = llm_pool_df[item_name].values  # <-- CORREÇÃO

        R_i = len(human_ratings_item)
        S_i = len(llm_ratings_item)

        if R_i == 0 or S_i == 0:
            continue

        item_weight = (R_i + S_i) / (R_total + S_total)

        disagreement_matrix = (human_ratings_item[None, :] - llm_ratings_item[:, None]) ** 2
        sum_disagreements_item = np.sum(disagreement_matrix)
        avg_item_disagreement = sum_disagreements_item / (R_i * S_i)

        total_observed_disagreement += item_weight * avg_item_disagreement

    d_o = total_observed_disagreement

    # --- 5. Calcular Métricas Finais (Eq. 6 e 13) ---
    if d_e == 0:
        kappa_x = 1.0 if d_o == 0 else 0.0
    else:
        # Adiciona um pequeno "print" para depuração
        print(f"Debug: d_o={d_o}, d_e={d_e}")
        kappa_x = 1.0 - (d_o / d_e)

    irr_product = human_irr * llm_irr
    if irr_product <= 0 or pd.isna(irr_product):
        normalized_kappa_x = np.nan
    else:
        normalized_kappa_x = kappa_x / np.sqrt(irr_product)

    return {
        'kappa_x': kappa_x,
        'normalized_kappa_x': normalized_kappa_x,
        'irr_human': human_irr,
        'irr_llm': llm_irr,
        'd_o': d_o,
        'd_e': d_e
    }


def calculate_xrr_human_vs_human(pool_X_ratings: pd.DataFrame,
                                 pool_Y_ratings: pd.DataFrame,
                                 pool_X_irr: float = None,
                                 pool_Y_irr: float = None) -> dict:
    """
    Calcula métricas xRR entre DOIS POOLS HUMANOS (ambos esparsos).
    Implementa κₓ (Eq. 9, 10) e κₓ normalizado (Eq. 13) para dados ausentes.

    Args:
        pool_X_ratings (pd.DataFrame): (N_raters_X x N_itens) com notas do Pool X.
        pool_Y_ratings (pd.DataFrame): (N_raters_Y x N_itens) com notas do Pool Y.
        pool_X_irr (float, optional): IRR pré-calculada do Pool X. Se None, calcula.
        pool_Y_irr (float, optional): IRR pré-calculada do Pool Y. Se None, calcula.

    Returns:
        dict: Dicionário com métricas xRR.
    """
    # --- 1. Preparar Pools e Calcular IRRs ---
    # Extrai os valores numpy, lidando com o 'Id' se presente
    pool_X_values = pool_X_ratings.drop('Id', axis=1, errors='ignore').values
    pool_Y_values = pool_Y_ratings.drop('Id', axis=1, errors='ignore').values

    # Calcula IRRs internos se não forem fornecidos
    if pool_X_irr is None:
        pool_X_irr = krippendorff.alpha(pool_X_values, level_of_measurement='interval')
    if pool_Y_irr is None:
        pool_Y_irr = krippendorff.alpha(pool_Y_values, level_of_measurement='interval')

    # --- 2. Calcular d_e (Desacordo Esperado - Eq. 10) ---
    # Coleta todas as notas válidas (não-NaN) de cada pool
    all_X_ratings = pool_X_values.flatten()
    all_X_ratings = all_X_ratings[~np.isnan(all_X_ratings)]  # Lida com esparso

    all_Y_ratings = pool_Y_values.flatten()
    all_Y_ratings = all_Y_ratings[~np.isnan(all_Y_ratings)]  # Lida com esparso

    R_total = len(all_X_ratings)  # R total (Eq. 10)
    S_total = len(all_Y_ratings)  # S total (Eq. 11)

    if R_total == 0 or S_total == 0:
        return {'kappa_x': np.nan, 'normalized_kappa_x': np.nan,
                'irr_X': pool_X_irr, 'irr_Y': pool_Y_irr,
                'd_o': np.nan, 'd_e': np.nan}

    # Cálculo eficiente de d_e (E[(X-Y)²] = E[X²] + E[Y²] - 2*E[X]*E[Y])
    mean_X = np.mean(all_X_ratings)
    mean_Y = np.mean(all_Y_ratings)
    mean_sq_X = np.mean(np.square(all_X_ratings))
    mean_sq_Y = np.mean(np.square(all_Y_ratings))
    d_e = mean_sq_X + mean_sq_Y - (2 * mean_X * mean_Y)

    # --- 3. Calcular d_o (Desacordo Observado - Eq. 9) ---
    total_observed_disagreement = 0

    # Garantir que os DataFrames estejam alinhados (só colunas de score)
    pool_X_scores_df = pool_X_ratings.drop('Id', axis=1, errors='ignore')
    pool_Y_scores_df = pool_Y_ratings.drop('Id', axis=1, errors='ignore')

    # Pegar todos os itens (colunas) de ambos
    all_items = set(pool_X_scores_df.columns) | set(pool_Y_scores_df.columns)

    for item_name in all_items:
        if item_name not in pool_X_scores_df.columns or item_name not in pool_Y_scores_df.columns:
            continue  # Item precisa estar em ambos os pools para comparar

        ratings_X_item = pool_X_scores_df[item_name].dropna().values
        ratings_Y_item = pool_Y_scores_df[item_name].dropna().values

        R_i = len(ratings_X_item)  # R(i) da Eq. 9
        S_i = len(ratings_Y_item)  # S(i) da Eq. 9

        if R_i == 0 or S_i == 0:
            continue  # Pula item se um dos pools não tiver dados para ele

        item_weight = (R_i + S_i) / (R_total + S_total)  # Ponderação (Eq. 9)

        # Média de todos os pares de desacordo para este item
        # Usamos broadcasting (rápido): (1, R_i) - (S_i, 1) -> (S_i, R_i)
        disagreement_matrix = (ratings_X_item[None, :] - ratings_Y_item[:, None]) ** 2
        sum_disagreements_item = np.sum(disagreement_matrix)
        avg_item_disagreement = sum_disagreements_item / (R_i * S_i)

        total_observed_disagreement += item_weight * avg_item_disagreement

    d_o = total_observed_disagreement

    # --- 4. Calcular Métricas Finais (Eq. 6 e 13) ---
    if d_e == 0:
        kappa_x = 1.0 if d_o == 0 else 0.0
    else:
        kappa_x = 1.0 - (d_o / d_e)

    irr_product = pool_X_irr * pool_Y_irr
    if irr_product <= 0 or pd.isna(irr_product):
        normalized_kappa_x = np.nan
    else:
        normalized_kappa_x = kappa_x / np.sqrt(irr_product)  # [cite: 213-216]

    return {
        'kappa_x': kappa_x,
        'normalized_kappa_x': normalized_kappa_x,
        'irr_X': pool_X_irr,
        'irr_Y': pool_Y_irr,
        'd_o': d_o,
        'd_e': d_e
    }

df_notas, df_respostas, df_mercado, df_bb, df_terceiro = load_and_clean_data()
LIMITE_LEGIVEL = 6.19 #mean
# LIMITE_LEGIVEL = 6.06 #median

print(df_mercado.to_string())
print(df_terceiro.to_string())
print(df_bb.to_string())
# print(df_respostas)

# print(df_notas.to_string())
plot_score_distribution(df_notas)
count_scores(df_notas)

group1, group2, group3 = split_random_groups(df_notas)
# print(group1.to_string())
# print(group2.to_string())
# print(group3.to_string())

df_devs = pd.concat([df_mercado, df_terceiro], axis=0, ignore_index=True).drop(['terceiro'], axis=1)

# Temos aqui todos os desenvolvedores respondentes
df_devs = pd.concat([df_devs, df_bb.drop(['job time'], axis=1)], axis=0, ignore_index=True)

print(df_devs.to_string())
# gerar_latex_tabelas_resumo(df_devs)
# gerar_latex_experiencia_por_cargo(df_devs)
# plotar_distribuicao_cargo_experiencia(df_devs)

# fazemos o merge dos 3 grupos de respostas, desta forma conseguimos realizar a análise dos dados
df_group = [pd.merge(group1, df_devs, on='Id', how='left'),
            pd.merge(group2, df_devs, on='Id', how='left'),
            pd.merge(group3, df_devs, on='Id', how='left')]

roles = ['Junior', 'Mid', 'Senior', 'Staff', 'Principal']
attribute = "role"

# roles = ['Menos de 2 anos', 'Entre 2 e 5 anos', 'Entre 5 e 10 anos', 'Entre 10 e 20 anos', 'Mais de 20 anos']
# attribute = "experience"

# roles = ['Java', 'Javascript/Typescript', 'C/C++', "Python", 'Outra']
# attribute = "language"


medias = {}
std = {}
statisticas_por_cluster(attribute)

# print(df_devs.to_string())

readable = pd.DataFrame()
overall_medias = []
overall_stds = []
exps = ['Menos de 2 anos', 'Entre 2 e 5 anos', 'Entre 5 e 10 anos', 'Entre 10 e 20 anos', 'Mais de 20 anos']
for df_group in df_group:
    print(df_group.to_string())

    alpha_results = calculate_krippendorff_alpha_by_group(df_group)
    print("Gerais do Krippendorff \t", alpha_results['Overall_Alpha'])
    alpha_results = calculate_krippendorff_alpha_by_group(df_group[df_group['language'] == 'Java'])
    print("--- Krippendorff de Java:\t ", alpha_results['Overall_Alpha'] )
    alpha_results = calculate_krippendorff_alpha_by_group(df_group[df_group['language'] == 'Python'])
    print("--- Krippendorff de Python:\t ", alpha_results['Overall_Alpha'] )
    alpha_results = calculate_krippendorff_alpha_by_group(df_group[df_group['language'] == 'C/C++'])
    print("--- Krippendorff de C/C++:\t ", alpha_results['Overall_Alpha'] )
    alpha_results = calculate_krippendorff_alpha_by_group(df_group[df_group['language'] == 'Javascript/Typescript'])
    print("--- Krippendorff de Javascript/Typescript:\t ", alpha_results['Overall_Alpha'] )

    for role in roles:
        alpha_results = calculate_krippendorff_alpha_by_group(df_group[df_group['role'] == role])
        print(f'--- Krippendorff de {role}:\t ', alpha_results['Overall_Alpha'] )

    for exp in exps:
        alpha_results = calculate_krippendorff_alpha_by_group(df_group[df_group['experience'] == exp])
        print(f'--- Krippendorff de {exp}:\t ', alpha_results['Overall_Alpha'] )

    # Drop non-score columns
    score_columns = df_group.drop(['Id', 'experience', 'language', 'role'], axis=1)
    if not score_columns.empty:
        overall_medias.append(score_columns.mean())
        overall_stds.append(score_columns.std())
if overall_medias:
    overall_means = pd.concat(overall_medias, axis=1).mean(axis=1)
    overall_std_devs = pd.concat(overall_stds, axis=1).mean(axis=1)
else:
    overall_means = pd.Series()
    overall_std_devs = pd.Series()

df_overall_means = pd.DataFrame(overall_means, columns=['Score All'])
df_overall_means.index.name = 'File'
readable.index = df_overall_means.index
readable['All'] = df_overall_means['Score All'].apply(lambda score: 1 if score >= LIMITE_LEGIVEL else 0)
df_overall_means.sort_index(inplace=True)
media = df_overall_means['Score All'].mean()
print("Média\n",media)
print("Mediana\n",df_overall_means['Score All'].median())
DESVIO_GLOBAL = df_overall_means['Score All'].std()
print("Desvio Padrão\n",DESVIO_GLOBAL)

for role in roles:
    medias[role] = medias[role].to_frame(name='score')
    readable[role] = medias[role]['score'].apply(lambda score: 1 if score >= LIMITE_LEGIVEL else 0)
    # print(role,' Medias\n',medias[role])
    std[role] = std[role].to_frame(name='std')
    # print(role,' Desvios\n',std[role])



df_llm = {}
df_omega = {}
llms = ['Gemini25flash-lite-thinking-10k', 'Gemini25flash-lite-thinking-1k', 'Gemini25flash-lite', 'Gemini20flash', 'Gemini25flash', 'Gemini25flash-thinking-10k',
        'Gemini20pro', 'Gemini25pro', 'Gemini30pro', 'Claude35-haiku', 'Claude37-sonnet', 'Claude45-sonnet', 'Claude45-sonnet-thinking', 'Claude45-haiku', 'Claude45-haiku-thinking',
        'Claude45-opus',  'Claude45-opus-thinking',
        'DeepSeek-V3', 'DeepSeek-V3.2-Exp', 'DeepSeek-V3.2-Exp-thinking-10k', "Kimi-K2-thinking", "Qwen3-Coder-480B-A35B-Instruct", "Qwen3-235B-A22B-2507-thinking",
        'Llama31-405b', 'Llama31-8b', 'Llama-4-Maverick-17B-128E-Instruct-FP8', 'Llama-4-Scout-17B-16E-Instruct-FP8', "grok-4-1-fast-thinking", "grok-4-0709-thinking",
        'GPT4o', 'GPT4o-mini', 'GPT-5-nano', 'GPT-5-nano-thinking', 'GPT-5-mini', 'GPT-5-mini-thinking', "GPT-5", "GPT-5-thinking", "GPT-51-thinking", "GPT-oss-120b-thinking"]
scenarios = ['Original', 'NoComments', 'BadNames']

for llm in llms:
    for scenario in scenarios:
        # print(f'Carregando {scenario} para {llm}')
        df_scenario = load(scenario, llm)
        # print(df_scenario)
        if llm not in df_llm or df_llm[llm].empty:
            df_llm[llm] = df_scenario
        else:
            df_llm[llm] = pd.concat([df_llm[llm], df_scenario], axis=0)
    df_llm[llm] = df_llm[llm].T
    # print(df_llm[llm].to_string())

    # OMEGA DE MACDONALD

    omega_geral = calculate_mcdonalds_omega_overall(df_llm[llm])
    interpretacao = interpret_reliability(omega_geral)
    # print(f"\nAnálise de Confiabilidade Geral para :\n")
    print(f"{llm} \t- (ω): \t\t{omega_geral:.4f}")
    # print(f"Interpretação: {interpretacao}")
    df_omega[llm] = omega_geral

# for llm in llms:
#     print(f'Omega de McDonald (ω) para {llm} = \t{df_omega[llm]}')

for llm in llms:
    medias[llm] = df_llm[llm].mean().to_frame(name='score')
    readable[llm] = medias[llm]['score'].apply(lambda score: 1 if score >= LIMITE_LEGIVEL else 0)
    std[llm] = df_llm[llm].std().to_frame(name='std')

# for item in std:
#     print(item, ' std\n', std[item])

# correlacoes = calcular_correlacoes_spearman(medias, 'score')
# for par, valores in correlacoes.items():
#     print(f"Correlação entre {par[0]} e {par[1]}:")
#     print(f"  Correlação: {valores['correlacao']:.4f}")
#     print(f"  P-valor: {valores['p_valor']:.4f}")
#     print("-" * 30)

df_sco_all = pd.DataFrame
medias_scenario = {}
for scenario in scenarios:
    dados_sco = load_sco_data(scenario)
    medias_scenario['SCO' + scenario] = dados_sco.sort_index()
    if df_sco_all.empty:
        df_sco_all = pd.DataFrame(medias_scenario['SCO' + scenario])
    else:
        df_sco_all = pd.concat([df_sco_all, medias_scenario['SCO' + scenario]], ignore_index=False)
    # print('\n', 'SCO ' + scenario, '\n', medias_scenario['SCO' + scenario])

medias['SCO'] = df_sco_all
# for item in medias:
#     print(item)
#     print(item, ' medias\n', medias[item])

readable['SCO'] = df_sco_all['score'].apply(lambda score: 1 if score >= LIMITE_LEGIVEL else 0)
readable.sort_index(inplace=True)
print("Classificação Binária\n", readable.to_string())
calculate_accuracy_and_roc(readable)

print("\nCalculando Precision, Recall e F1-Score, com All as ground truth:")
metrics_results = calculate_p_r_f1(readable, readable['All'])
print(readable.to_string())

# metrica = ['F1-score', 'Precision', 'Recall']
# plot_binary_metrics(metrics_results, metrica[2])

# for scenario in scenarios:
#     for item in medias:
#         filtro = medias[item].index.str.startswith(scenario)
#         medias_scenario[item+scenario] = medias[item][filtro].sort_index()
#         print('\n',item+' '+scenario,'\n',medias_scenario[item+scenario])

# medias_por_scenario = {}
# for scenario in scenarios:
#     medias_por_scenario[scenario] = {}
#     for item in medias_scenario:
#         if item.endswith(scenario):
#             medias_por_scenario[scenario][item] = medias_scenario[item]
#
# for scenario in scenarios:
#     print(scenario, '-'*50)
#     for item in medias_por_scenario[scenario]:
#         print(item, ':\n',medias_por_scenario[scenario][item])

# correlacao_por_scenario = {}
# for scenario in scenarios:
#     correlacoes = calcular_correlacoes_spearman(medias_por_scenario[scenario], 'score')
#     correlacao_por_scenario[scenario] = correlacoes
#     for par, valores in correlacoes.items():
#         print(f"Correlação entre {par[0]} e {par[1]}: Correlação: {valores['correlacao']:.4f} P-valor: {valores['p_valor']:.4f}")
#         # print("-" * 30)
#     criar_heatmap_correlacoes(scenario, correlacoes)
#     criar_graficos_dispersao(scenario, medias_por_scenario[scenario])
#     criar_boxplot_lado_a_lado(scenario, medias_por_scenario[scenario])
#
#     wilcoxon_test_result = calcular_teste_wilcoxon(medias_por_scenario[scenario], 'score')
#     for par, valores in wilcoxon_test_result.items():
#         print(f"Wilcoxon entre {par[0]} e {par[1]}: Estatistica: {valores['estatistica']:.4f} P-valor: {valores['p_valor']:.4f}")

# correlacoes = calcular_correlacoes_spearman_all(medias, 'score')
# for par, valores in correlacoes.items():
#     print(f"Correlação entre {par[0]} e {par[1]}: Correlação: {valores['correlacao']:.4f} P-valor: {valores['p_valor']:.4f}")
#     # print("-" * 30)
# criar_heatmap_correlacoes('All', correlacoes)
# criar_graficos_dispersao('All', medias)
# criar_boxplot_lado_a_lado('All', medias)
#
# wilcoxon_test_result = calcular_teste_wilcoxon_all(medias, 'score')
# for par, valores in wilcoxon_test_result.items():
#     print(f"Wilcoxon entre {par[0]} e {par[1]}: Estatistica: {valores['estatistica']:.4f} P-valor: {valores['p_valor']:.4f}")

df_overall_means = df_overall_means.rename(
    columns={'Score All': 'score'}
)

medias_novas = medias
# print('MEDIAS')
# print(medias)
# print('df_overall_means\n',df_overall_means)

medias_novas['All'] = df_overall_means
df_medias = combine_lists_to_dataframe(medias_novas)
print("DF_MEDIAS\n",df_medias.to_string())

df_overall_std_devs = pd.DataFrame(overall_std_devs, columns=['std'])
df_overall_std_devs.index.name = 'File'
std_novas = std
std_novas['All'] = df_overall_std_devs
df_desvio_padrao = combine_lists_to_dataframe_std(std_novas)
print("\n\nDF_DESVIO_PADRAO\n", df_desvio_padrao.to_string())


classificacao_ternaria = classify_ternary_around_mean(df_medias, LIMITE_LEGIVEL, DESVIO_GLOBAL, 1)
print("Classificação Ternária\n", classificacao_ternaria.to_string())

print("\nCalculando Precision, Recall e F1-Score, com All as ground truth para Classificação Ternária:")
metrics_results = calculate_multiclass_metrics(classificacao_ternaria, classificacao_ternaria['All'])
display_results_like_paper(metrics_results)
# plot_ternary_metrics(metrics_results, metrica[2])


# for scenario in scenarios:
#     plot_java_file_scores(df_medias[df_medias.index.str.startswith('E1')].sort_index(), scenario + '-1')
#     plot_java_file_scores(df_medias[df_medias.index.str.startswith('E2')].sort_index(), scenario + '-2')
#     plot_java_file_scores(df_medias[df_medias.index.str.startswith('E3')].sort_index(), scenario + '-3')
#     plot_java_file_scores(df_medias[df_medias.index.str.startswith('E4')].sort_index(), scenario + '-4')


# compara_matrizes_correlacao()

print("\n" + "=" * 80)
print("--- 🚀 Calculando Métricas xRR (Humanos vs LLMs) ---")
print("=" * 80)

xrr_results = {}

# 1. Calcular IRR_Human (Alpha de Krippendorff) para todos os 123 humanos nos 30 itens
#    A variável 'df_notas' (123x30, esparsa) deve estar disponível globalmente.

df_llm['SCO'] = df_sco_all['score'].to_frame(name='score').T
df_omega['SCO'] = 1
llms.append("SCO")

try:
    human_pool_for_irr = df_notas.drop('Id', axis=1, errors='ignore').values
    irr_human_overall = krippendorff.alpha(human_pool_for_irr, level_of_measurement='interval')
    print(f"IRR Humano (Alpha) Geral (N=123): {irr_human_overall:.4f}\n")
except Exception as e:
    print(f"ERRO: Não foi possível calcular o IRR Humano. Verifique 'df_notas'. Erro: {e}")
    irr_human_overall = None

if irr_human_overall is not None:
    # 2. Iterar sobre cada LLM e calcular o xRR
    for llm in llms:
        if llm not in df_omega:
            print(f"Aviso: Omega (IRR_LLM) não encontrado para {llm}. Pulando xRR.")
            print("       -> Certifique-se de DESCOMENTAR o bloco 'OMEGA DE MACDONALD'.")
            continue

        if llm not in df_llm:
            print(f"Aviso: Dados de notas (df_llm) não encontrados para {llm}. Pulando.")
            continue

        irr_llm = df_omega[llm]  # IRR_LLM (Omega)
        df_llm_ratings = df_llm[llm]  # Pool LLM (30 itens x 10 runs)

        # Calcular xRR:
        # Pool Humano = df_notas (123 raters x 30 items)
        # Pool LLM    = df_llm_ratings (30 items x 10 runs)
        # IRR Humano  = irr_human_overall (Alpha)
        # IRR LLM     = irr_llm (Omega)

        xrr = calculate_xrr_metrics(df_notas,
                                    df_llm_ratings,
                                    llm_irr=irr_llm,
                                    human_irr=irr_human_overall)

        xrr_results[llm] = xrr
        print(f"  Resultados para: {llm}")
        print(f"    ├─ κₓ (Cross-Kappa):       {xrr['kappa_x']:.4f}")
        print(f"    └─ Normalized κₓ (Similar.): {xrr['normalized_kappa_x']:.4f}")

    # 3. Exibir tabela de classificação
    if xrr_results:
        df_xrr = pd.DataFrame.from_dict(xrr_results, orient='index')
        df_xrr_sorted = df_xrr.sort_values(by='normalized_kappa_x', ascending=False)

        print("\n" + "-" * 80)
        print("--- Tabela de Resultados xRR (Ordenado por Similaridade com Humanos) ---")
        print(df_xrr_sorted[['normalized_kappa_x', 'kappa_x', 'irr_llm']].to_string(float_format="%.4f"))
        print(f"\n(IRR Humano de referência: {irr_human_overall:.4f})")
        print("=" * 80 + "\n")

print("\n" + "=" * 80)
print("--- 🚀 Calculando Métricas xRR (Humanos Overall vs. Roles) ---")
print("=" * 80)

# 1. Criar DataFrame mestre com todos os dados humanos (notas + demografia)
# df_notas (123x30) + df_devs (123xN)
# As colunas de 'df_devs' são ['Id', 'experience', 'language', 'role']
df_human_master = pd.merge(df_notas, df_devs, on='Id', how='left')

# 2. Definir Pool X (Overall)
pool_X_ratings = df_notas  # (123x30, esparso)

# Busca o IRR overall calculado no bloco anterior (Humanos vs LLMs)
if 'irr_human_overall' not in locals():
    print("Calculando IRR Humano Overall (Pool X)...")
    irr_human_overall = krippendorff.alpha(pool_X_ratings.drop('Id', axis=1, errors='ignore').values, level_of_measurement='interval')

pool_X_irr = irr_human_overall
print(f"IRR (Alpha) do Pool X (Overall, N=123): {pool_X_irr:.4f}\n")

# 3. Iterar e calcular xRR para cada Role (Pool Y)
roles_to_analyze = ['Junior', 'Mid', 'Senior', 'Staff', 'Principal']
xrr_role_results = {}

for role in roles_to_analyze:
    # 3a. Preparar Pool Y (Dados do Role)
    # Filtra do DataFrame mestre para obter os raters daquele role
    df_role_raters = df_human_master[df_human_master['role'] == role]

    # Isola apenas as colunas de score para o Pool Y
    # Mantém apenas as colunas de score originais de df_notas
    score_cols = df_notas.columns.drop('Id', errors='ignore')
    pool_Y_ratings = df_role_raters[score_cols]  # (N_role x 30, esparso)

    if pool_Y_ratings.empty or pool_Y_ratings.shape[0] < 2:
        print(f"Pulando {role}: Faltam avaliadores (N={pool_Y_ratings.shape[0]})")
        continue

    print(f"--- Comparando Overall vs. {role} (N={pool_Y_ratings.shape[0]}) ---")

    # 3b. Calcular xRR usando a nova função (Esparso vs Esparso)
    xrr = calculate_xrr_human_vs_human(
        pool_X_ratings,  # Pool X (Overall)
        pool_Y_ratings,  # Pool Y (Role)
        pool_X_irr  # IRR de X (Overall)
        # A função irá calcular o pool_Y_irr internamente
    )

    xrr_role_results[role] = xrr
    print(f"  ├─ IRR (Alpha) do {role}: {xrr['irr_Y']:.4f}")
    print(f"  ├─ κₓ (Cross-Kappa):       {xrr['kappa_x']:.4f}")
    print(f"  └─ Normalized κₓ (Similar.): {xrr['normalized_kappa_x']:.4f}")

# 4. Exibir tabela de classificação
if xrr_role_results:
    df_xrr_roles = pd.DataFrame.from_dict(xrr_role_results, orient='index')
    # Renomeia colunas para clareza
    df_xrr_roles.columns = ['kappa_x', 'normalized_kappa_x', 'irr_Overall', 'irr_Role', 'd_o', 'd_e']
    df_xrr_roles_sorted = df_xrr_roles.sort_values(by='normalized_kappa_x', ascending=False)

    print("\n" + "-" * 80)
    print("--- Tabela de Resultados xRR (Overall vs. Roles) ---")
    print(df_xrr_roles_sorted[['normalized_kappa_x', 'kappa_x', 'irr_Role']].to_string(float_format="%.4f"))
    print(f"\n(IRR Overall de referência: {pool_X_irr:.4f})")
    print("=" * 80 + "\n")