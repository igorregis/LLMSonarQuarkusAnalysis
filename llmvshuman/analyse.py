import pandas as pd
import os

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
    if dados is not None:
        print(f"Dados carregados com sucesso!")
        print(dados)  # Exibe as primeiras linhas do DataFrame
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

    return df_respostas, df_mercado, df_bb, df_terceiro


df_respostas, df_mercado, df_bb, df_terceiro = load_and_clean_data()

print(df_mercado.to_string())
print(df_terceiro.to_string())
print(df_bb.to_string())
print(df_respostas)