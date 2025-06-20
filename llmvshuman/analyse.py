import re
from sklearn.metrics import accuracy_score, roc_curve, auc
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
from numpy.ma.extras import column_stack


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
            plt.savefig(f'graficos/roc-{column}.png', dpi=300)

    # Print accuracy results
    print("Accuracy Results:")
    for column, metrics in results.items():
        print(f"{column}: Accuracy = {metrics['accuracy']:.4f}")

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
            file_name = re.sub(r'^BadNames-AnoBissexto', 'B1-AnoBissexto', file_name)
            file_name = re.sub(r'^BadNames-CalculaAreaCirculo', 'B1-CalculaAreaCirculo', file_name)
            file_name = re.sub(r'^BadNames-CalculaAreaTrianguloIsoceles', 'B1-CalculaAreaTrianguloIsoceles', file_name)
            file_name = re.sub(r'^BadNames-GetMedia', 'B12-GetMedia', file_name)
            file_name = re.sub(r'^BadNames-MonthSubtrai', 'B2-MonthSubtrai', file_name)
            file_name = re.sub(r'^BadNames-MonthPlus', 'B2-MonthPlus', file_name)
            file_name = re.sub(r'^BadNames-MonthPrimeiroMesDoTrimestre', 'B2-MonthPrimeiroMesDoTrimestre', file_name)
            file_name = re.sub(r'^BadNames-Combine', 'B2-Combine', file_name)
            file_name = re.sub(r'^BadNames-CarregarFilhos', 'B3-CarregarFilhos', file_name)
            file_name = re.sub(r'^BadNames-CalculaDVBase10', 'B3-CalculaDVBase10', file_name)

            file_name = re.sub(r'^NoComments-AnoBissexto', 'N1-AnoBissexto', file_name)
            file_name = re.sub(r'^NoComments-CalculaAreaCirculo', 'N1-CalculaAreaCirculo', file_name)
            file_name = re.sub(r'^NoComments-CalculaAreaTrianguloIsoceles', 'N1-CalculaAreaTrianguloIsoceles', file_name)
            file_name = re.sub(r'^NoComments-GetMedia', 'N12-GetMedia', file_name)
            file_name = re.sub(r'^NoComments-MonthSubtrai', 'N2-MonthSubtrai', file_name)
            file_name = re.sub(r'^NoComments-MonthPlus', 'N2-MonthPlus', file_name)
            file_name = re.sub(r'^NoComments-MonthPrimeiroMesDoTrimestre', 'N2-MonthPrimeiroMesDoTrimestre', file_name)
            file_name = re.sub(r'^NoComments-Combine', 'N2-Combine', file_name)
            file_name = re.sub(r'^NoComments-CarregarFilhos', 'N3-CarregarFilhos', file_name)
            file_name = re.sub(r'^NoComments-CalculaDVBase10', 'N3-CalculaDVBase10', file_name)

            file_name = re.sub(r'^Original-AnoBissexto', 'O1-AnoBissexto', file_name)
            file_name = re.sub(r'^Original-CalculaAreaCirculo', 'O1-CalculaAreaCirculo', file_name)
            file_name = re.sub(r'^Original-CalculaAreaTrianguloIsoceles', 'O1-CalculaAreaTrianguloIsoceles', file_name)
            file_name = re.sub(r'^Original-GetMedia', 'O12-GetMedia', file_name)
            file_name = re.sub(r'^Original-MonthSubtrai', 'O2-MonthSubtrai', file_name)
            file_name = re.sub(r'^Original-MonthPlus', 'O2-MonthPlus', file_name)
            file_name = re.sub(r'^Original-MonthPrimeiroMesDoTrimestre', 'O2-MonthPrimeiroMesDoTrimestre', file_name)
            file_name = re.sub(r'^Original-Combine', 'O2-Combine', file_name)
            file_name = re.sub(r'^Original-CarregarFilhos', 'O3-CarregarFilhos', file_name)
            file_name = re.sub(r'^Original-CalculaDVBase10', 'O3-CalculaDVBase10', file_name)
            if file_name not in combined_data:
                combined_data[file_name] = {}
            combined_data[file_name][name] = row['score']

    return pd.DataFrame(combined_data).T

df_notas, df_respostas, df_mercado, df_bb, df_terceiro = load_and_clean_data()
LIMITE_LEGIVEL = 6.19 #mean
# LIMITE_LEGIVEL = 6.06 #median

# print(df_mercado.to_string())
# print(df_terceiro.to_string())
# print(df_bb.to_string())
# print(df_respostas)

# print(df_notas.to_string())
plot_score_distribution(df_notas)
count_scores(df_notas)

group1, group2, group3 = split_random_groups(df_notas)
# print(group1.to_string())
# print(group2.to_string())
# print(group3.to_string())

df_devs = pd.concat([df_mercado, df_terceiro], axis=0, ignore_index=True).drop(['terceiro'], axis=1)
df_devs = pd.concat([df_devs, df_bb.drop(['job time'], axis=1)], axis=0, ignore_index=True)

print(df_devs.to_string())

df_group = [pd.merge(group1, df_devs, on='Id', how='left'),
            pd.merge(group2, df_devs, on='Id', how='left'),
            pd.merge(group3, df_devs, on='Id', how='left')]

roles = ['Junior', 'Mid', 'Senior', 'Staff', 'Principal']

medias = {}
std = {}
for role in roles:
    role_medias = []
    role_stds = []
    for group in df_group:
        # print(group)
        role_group = group[group['role'] == role].drop(['Id', 'experience', 'language', 'role'], axis=1)
        if not role_group.empty:
            role_medias.append(role_group.mean())
            role_stds.append(role_group.std())
    if role_medias:
        medias[role] = pd.concat(role_medias, axis=1).mean(axis=1)
        std[role] = pd.concat(role_stds, axis=1).mean(axis=1)
    else:
        medias[role] = pd.Series()
        std[role] = pd.Series()

readable = pd.DataFrame()
overall_medias = []
overall_stds = []
for df_group in df_group:
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
print(df_overall_means['Score All'].mean())
print(df_overall_means['Score All'].median())

for role in roles:
    medias[role] = medias[role].to_frame(name='score')
    readable[role] = medias[role]['score'].apply(lambda score: 1 if score >= LIMITE_LEGIVEL else 0)
    # print(role,' Medias\n',medias[role])
    std[role] = std[role].to_frame(name='std')
    # print(role,' Desvios\n',std[role])

df_llm = {}
llms = ['Gemini20flash', 'Gemini20pro', 'Claude35-haiku', 'Claude37-sonnet', 'DeepSeek-V3', 'Llama31-405b', 'Llama31-8b', 'GPT4o', 'GPT4o-mini']
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
print(readable.to_string())
calculate_accuracy_and_roc(readable)

for scenario in scenarios:
    for item in medias:
        filtro = medias[item].index.str.startswith(scenario)
        medias_scenario[item+scenario] = medias[item][filtro].sort_index()
        # print('\n',item+' '+scenario,'\n',medias_scenario[item+scenario])

medias_por_scenario = {}
for scenario in scenarios:
    medias_por_scenario[scenario] = {}
    for item in medias_scenario:
        if item.endswith(scenario):
            medias_por_scenario[scenario][item] = medias_scenario[item]

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

df_medias = combine_lists_to_dataframe(medias)
print(df_medias.to_string())

for scenario in scenarios:
    filter = 'O'
    if scenario == 'BadNames':
        filter = 'B'
    elif scenario == 'NoComments':
        filter = 'N'

    plot_java_file_scores(df_medias[df_medias.index.str.startswith(filter + '1')].sort_index(), scenario + '-1')
    plot_java_file_scores(df_medias[df_medias.index.str.startswith(filter+'2')].sort_index(), scenario + '-2')
    plot_java_file_scores(df_medias[df_medias.index.str.startswith(filter + '3')].sort_index(), scenario + '-3')


# compara_matrizes_correlacao()