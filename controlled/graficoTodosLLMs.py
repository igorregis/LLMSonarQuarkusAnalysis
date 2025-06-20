from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import multiprocessing
import scipy.stats as stats
import random
import seaborn as sns
import itertools
from collections import defaultdict
import pandas as pd
import json
import numpy as np
from scipy.stats import binomtest, shapiro, levene, norm
from statsmodels.sandbox.stats.multicomp import multipletests
import os
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
import pingouin as pg

def main():
    print(os.getcwd())

    # Defina as listas de dados e seus respectivos diretórios
    data_lists = {
        'clean_code': 'CleanCode',
        # 'bad_names_no_comments': 'BadNamesNoComments',
        'bad_names': 'BadNames',
        'no_comments': 'NoComments',
        'original': 'Original'
    }
    llm_lists = {
        'GPT4o': 'GPT4o',
        'Claude35_sonnet': 'Claude35-sonnet',
        'Claude37_sonnet': 'Claude37-sonnet',
        'Gemini15pro': 'Gemini15pro',
        'Gemini20pro': 'Gemini20pro',
        'Llama31_405b': 'Llama31-405b',
        'GPT4o_mini': 'GPT4o-mini',
        'Claude35_haiku': 'Claude35-haiku',
        'Gemini15flash': 'Gemini15flash',
        'Gemini20flash': 'Gemini20flash',
        'Llama31_8b': 'Llama31-8b',
        'DeepSeek_V3': 'DeepSeek-V3'

    }

    data_lists_copy = data_lists.copy()
    df = {}
    for llm_list, llm in llm_lists.items():
        # Itere sobre cada lista de dados e diretório
        for data_list, directory in data_lists_copy.items():
            # Inicialize a lista de dados
            data_lists[llm_list + '-' + data_list] = []
            # Carregue os dados de 10 arquivos no diretório
            for i in range(1, 11):
                file_name = f'{llm}/{directory[0] + directory[1:]}/controlled{directory}{llm}-{i}.json'
                # Carregue os dados do arquivo
                loadData(data_lists[llm_list + '-' + data_list], file_name)

                # Converter a lista em um DataFrame
                df[llm_list + '-' + data_list] = pd.DataFrame(data_lists[llm_list + '-' + data_list], columns=['name',
                                                                                                               'score',
                                                                                                               'code_smells',
                                                                                                               'cognitive_complexity',
                                                                                                               'comment_lines_density',
                                                                                                               'complexity',
                                                                                                               'lines',
                                                                                                               'sqale_rating',
                                                                                                               'statements'])
            # if data_list == 'no_comments' and llm_list == 'Gemini15flash':
            #     print(f'{data_list}-{llm_list}---------------------------------\n',data_lists[llm_list + '-' + data_list])
            #     temp = data_lists[llm_list + '-' + data_list].copy()
            #     temp.sort(key=lambda x: x[0])
            #     for item in temp:
            #         print(item[0], item[1])
            if data_list == 'clean_code':
                clean_code = data_list
                print(f'\nEstatisticas para {llm_list + "-" + data_list}\n')
                calcular_estatisticas(df[llm_list + '-' + data_list])
                # calcular_variacao(df[llm_list + '-' + data_list])

            if data_list == 'original':
                original = data_list
                print(f'\nEstatisticas para {llm_list + "-" + data_list}\n')
                calcular_estatisticas(df[llm_list + '-' + data_list])
                # calcular_variacao(df[llm_list + '-' + data_list])

            if data_list == 'bad_names':
                bad_names = data_list
                print(f'\nEstatisticas para {llm_list + "-" + data_list}\n')
                calcular_estatisticas(df[llm_list + '-' + data_list])
                # calcular_variacao(df[llm_list + '-' + data_list])

            if data_list == 'no_comments':
                no_comments = data_list
                print(f'\nEstatisticas para {llm_list + "-" + data_list}\n')
                calcular_estatisticas(df[llm_list + '-' + data_list])
                # calcular_variacao(df[llm_list + '-' + data_list])

            # if data_list == 'bad_names_no_comments':
            #     bad_names_no_comments = data_list
            #     print(f'\nEstatisticas para {llm_list + "-" + data_list}\n')
            #     calcular_estatisticas(df[llm_list + '-' + data_list])
            #     calcular_variacao(df[llm_list + '-' + data_list])

    for data in data_lists_copy.keys():
        gerar_boxplot(data_lists, llm_lists, data)

    gerar_boxplot_sobreposto(data_lists, llm_lists, original, no_comments)
    gerar_boxplot_sobreposto(data_lists, llm_lists, original, bad_names)
    # gerar_boxplot_sobreposto(data_lists, llm_lists, original, bad_names_no_comments)
    gerar_boxplot_sobreposto(data_lists, llm_lists, original, clean_code)

    classifier_data = load_classifier_data()
    print(f'classifier_data\n{classifier_data}')

    # teste_geral_mannwhitneyu(data_lists)
    # print(df)

    df_geral = criar_data_frame_geral(data_lists)
    # print('data_list--------------------------------------\n', data_lists)
    # print('df_geral------------------------------------\n', df_geral.to_string())

    # matriz_correlacao_geral(df_geral)
    # print(df_geral.to_string())
                # 'bad_names_no_comments',
    modelos = ['GPT4o', 'GPT4o_mini', 'Gemini20pro', 'Gemini20flash', 'Llama31_405b', 'Llama31_8b', 'Claude37_sonnet', 'Claude35_haiku', 'DeepSeek_V3']
    cenarios = ['original', 'bad_names',
                'clean_code', 'no_comments']
    classes = ['DoubleSummaryStatistics.java', 'Month.java', 'DynamicTreeNode.java', 'ElementTreePanel.java', 'HelloWorld.java', 'Notepad.java',
               'SampleData.java', 'SampleTree.java', 'SampleTreeCellRenderer.java', 'SampleTreeModel.java', 'Stylepad.java', 'Wonderland.java']
    class_codes = {'DoubleSummaryStatistics.java':'C1', 'Month.java':'C2', 'DynamicTreeNode.java':'C3', 'ElementTreePanel.java':'C4', 'HelloWorld.java':'C5', 'Notepad.java':'C6',
                   'SampleData.java':'C7', 'SampleTree.java':'C8', 'SampleTreeCellRenderer.java':'C9', 'SampleTreeModel.java':'C10', 'Stylepad.java':'C11', 'Wonderland.java':'C12'}

    # correlation_sco_llms(cenarios, class_codes, classifier_data, df_geral, llm_lists)

    # tabela_std(cenarios, class_codes, classes, df_geral, modelos)
    # tabela_mean(cenarios, class_codes, classes, df_geral, modelos)
    # concordancia(cenarios, class_codes, classes, df_geral, modelos)
    dfs = concordancia_df(cenarios, class_codes, classes, df_geral, modelos)
    dfs_with_sco = add_sco_column(dfs, classifier_data, class_codes)
    dfs_classified = apply_classification_rules(dfs_with_sco)

    # agreement_analysis = analyze_agreement(dfs_classified)
    # agreement_analysis = analyze_agreement_kappa(dfs_classified)
    # print_agreement_description(agreement_analysis)
    # print_latex_tables(agreement_analysis)

    agreement_analysis = analyze_precision_recall_f1(dfs_classified)
    print_metrics_description(agreement_analysis)
    print_prf_latex_tables(agreement_analysis)
    for df in dfs_classified:
        print('Scenario: ',df,'\n' , dfs_classified[df].to_string())


    # wilcoxon_cross_cenarios(df_geral)

    # print(classifier_data.info())
    # print(df_geral.to_string())


    # calcular_variancia_df(df_filtrado)
    # matriz_correlacao_geral(df_filtrado)
    # plotar_scatter(df_filtrado, 'Gemini15pro', 'comment_lines_density')
    # plotar_histogramas(df_geral)


def correlation_sco_llms(cenarios, class_codes, classifier_data, df_geral, llm_lists):
    # Prepare data for corr analysis
    df_cenarios = {}
    for cenario in cenarios:
        for classe, alias in class_codes.items():
            for llm, llm_list in llm_lists.items():
                classifier_data.loc[f'{cenario}-{classe}', llm] = \
                df_geral[df_geral.index.str.endswith(f'{cenario}-{classe}')][llm].mean()
        df_cenarios[cenario] = classifier_data[classifier_data.index.str.startswith(cenario)]
        print(df_cenarios[cenario].to_string())

        for llm, llm_list in llm_lists.items():
            correlacao = df_cenarios[cenario]['sco'].corr(df_cenarios[cenario][llm], method='spearman')
            print(f'{cenario}: SCO com {llm}', correlacao)
        teste_wilcoxon(df_cenarios[cenario])
    print(classifier_data.to_string())

def calcular_cliff_delta(coluna1, coluna2):
    return pg.compute_effsize(coluna1, coluna2, eftype='cles')

def calcular_r_wilcoxon(estatistica, n):
    z = (estatistica - (n * (n + 1) / 4)) / ((n * (n + 1) * (2 * n + 1) / 24) ** 0.5)
    r = z / (n ** 0.5)
    return r

def teste_wilcoxon(df):
    df = df.copy().dropna()
    colunas = len(df.columns)
    for i in range(colunas):
        for j in range(i + 1, colunas):
            coluna1 = df.iloc[:,i]
            coluna2 = df.iloc[:,j]
            estatistica, valor_p = wilcoxon(coluna1.values, coluna2.values)
            n = len(coluna1)  # Tamanho da amostra
            cliff_delta = calcular_cliff_delta(coluna1, coluna2)
            r_wilcoxon = calcular_r_wilcoxon(estatistica, n)
            print(
                f'Wilcoxon de {df.columns[i]} com {df.columns[j]} - estatistica:{estatistica}, valor_p:, {valor_p}, Cliff\'s Delta: {cliff_delta}, r: {r_wilcoxon}')

def load_classifier_data():
    original = loadScalabrino('ReadabilityClassifier/Original.csv')
    original.columns = ['name', 'sco']
    # original = original.set_index('name')
    print(f'original\n{original}')

    noComments = loadScalabrino('ReadabilityClassifier/NoComments.csv')
    noComments.columns = ['name', 'sco']
    # noComments = noComments.set_index('name')
    print(f'noComments\n{noComments}')

    badNames = loadScalabrino('ReadabilityClassifier/BadNames.csv')
    badNames.columns = ['name', 'sco']
    # badNames = badNames.set_index('name')
    print(f'badNames\n{badNames}')

    cleanCode = loadScalabrino('ReadabilityClassifier/CleanCode.csv')
    cleanCode.columns = ['name', 'sco']
    # cleanCode = cleanCode.set_index('name')
    print(f'cleanCode\n{cleanCode}')

    original['name'] = 'original-' + original['name']
    original['sco'] = original['sco'] * 100
    noComments['name'] = 'no_comments-' + noComments['name']
    noComments['sco'] = noComments['sco'] * 100
    badNames['name'] = 'bad_names-' + badNames['name']
    badNames['sco'] = badNames['sco'] * 100
    cleanCode['name'] = 'clean_code-' + cleanCode['name']
    cleanCode['sco'] = cleanCode['sco'] * 100
    df_final = pd.concat([original, noComments, badNames, cleanCode], ignore_index=True)
    df_final = df_final.set_index('name')

    return df_final

def loadScalabrino(file_path):
    df = pd.read_csv(file_path, header=None)
    return df

def wilcoxon_cross_cenarios(df_geral):
    modelos = ['GPT4o', 'GPT4o_mini', 'Gemini20pro', 'Gemini20flash', 'Llama31_405b', 'Llama31_8b', 'Claude37_sonnet', 'Claude35_haiku', 'DeepSeek_V3']
    cenarios = [
                'original', 'bad_names',
                # 'original', 'bad_names_no_comments',
                'original', 'clean_code',
                'original', 'no_comments']
    classes = ['DoubleSummaryStatistics.java', 'Month.java', 'DynamicTreeNode.java', 'ElementTreePanel.java', 'HelloWorld.java', 'Notepad.java',
               'SampleData.java', 'SampleTree.java', 'SampleTreeCellRenderer.java', 'SampleTreeModel.java', 'Stylepad.java', 'Wonderland.java']
    class_codes = {'DoubleSummaryStatistics.java':'C1', 'Month.java':'C2', 'DynamicTreeNode.java':'C3', 'ElementTreePanel.java':'C4', 'HelloWorld.java':'C5', 'Notepad.java':'C6',
                   'SampleData.java':'C7', 'SampleTree.java':'C8', 'SampleTreeCellRenderer.java':'C9', 'SampleTreeModel.java':'C10', 'Stylepad.java':'C11', 'Wonderland.java':'C12'}
    resultados_ordenados = []
    alpha = 0.05
    for classe in classes:
        for modelo in modelos:
            for i in range(0, len(cenarios), 2):
                cenario1 = cenarios[i]
                cenario2 = cenarios[i+1]
                try:
                    grupo1 = cenario1 +"-"+ class_codes[classe] +"-"+ modelo
                    valores_grupo1 = df_geral[df_geral.index.str.contains(cenario1 + '-' + classe)][modelo].values.astype(int)
                    grupo2 = cenario2 +"-"+ class_codes[classe] +"-"+ modelo
                    valores_grupo2 = df_geral[df_geral.index.str.contains(cenario2 + '-' + classe)][modelo].values.astype(int)
                    # print(f'{cenario1}-{classe}-{modelo} : {valores_grupo1}')
                    # print(f'{cenario2}-{classe}-{modelo} : {valores_grupo2}')
                except ValueError as e:
                    print(f'Erro {df_geral[df_geral.index.str.contains(cenario2 + '-' + classe)][modelo]}')
                    print(e)
                n1 = len(valores_grupo1)
                n2 = len(valores_grupo2)
                n_total = n1 + n2
                try:
                    # stat, p_value = mannwhitneyu(valores_grupo1, valores_grupo2)
                    stat, p_value = wilcoxon(valores_grupo1, valores_grupo2)
                except ValueError as e:
                    print(f"Erro ao comparar {grupo1} e {grupo2}: {e}")
                    print("Verifique se os grupos têm dados suficientes e se os dados são numéricos.")
                    return {}

                resultados_ordenados.append({
                    "grupo1": grupo1,
                    "grupo2": grupo2,
                    "stat": stat,
                    "p_value": p_value,
                    'n_total': n_total,
                    'valores_grupo1': valores_grupo1,
                    'valores_grupo2': valores_grupo2
                })

    p_valores = [item['p_value'] for item in resultados_ordenados]
    reject, pvals_corrigidos, _, _ = multipletests(p_valores, alpha=alpha, method='bonferroni')

    # resultados_ordenados.sort(key=lambda x: x["grupo1"], reverse=True)

    for i, item in enumerate(resultados_ordenados):
        p_value = item['p_value']
        z = norm.ppf(1 - item['p_value'] / 2)
        item['r'] = z / np.sqrt(item['n_total'])
        if cenarios[3] in item['grupo2'] and modelos[8] in item['grupo2']:
            if p_value < (alpha / 10):
                mensagem_significancia = "Diferença SIGNIFICATIVA"
            elif p_value < alpha:
                mensagem_significancia = "Diferença MODERADAMENTE SIGNIFICATIVA"
            else:
                mensagem_significancia = "Diferença NÃO significativa"
            # if 'MODERADAMENTE' in mensagem_significancia:
            mediana_diff, li, ls = calcular_mediana_diferenca_bootstrap(item['valores_grupo1'], item['valores_grupo2'])
            if not np.array_equal(item['valores_grupo1'], item['valores_grupo2']):
                poder = calcular_poder_wilcoxon_paralelo(item['valores_grupo1'], item['valores_grupo2'], alpha)
                mensagem_significancia += f", poder {poder} e Mediana da Diferença = {mediana_diff:.2f} (IC 99%: {li:.2f}, {ls:.2f})"
            print(
                f"Wilcoxon Signed Rank test de {item['grupo1']} vs {item['grupo2']} (U={item['stat']:.2f}, p={p_value:.3e}), Tam. Efeito={item['r']:.2f}: {mensagem_significancia}")

def calcular_poder_wilcoxon_paralelo(grupo1, grupo2, alpha=0.05, num_simulacoes=1000, num_processos=None):
    """Calcula o poder do teste de Wilcoxon usando processamento paralelo."""

    if num_processos is None:
        num_processos = multiprocessing.cpu_count()  # Usa todos os núcleos disponíveis

    with multiprocessing.Pool(processes=num_processos) as pool:
        resultados = pool.starmap(simular_teste_wilcoxon, [(grupo1, grupo2, alpha)] * num_simulacoes)

    poder = sum(resultados) / num_simulacoes
    return poder

def simular_teste_wilcoxon(grupo1, grupo2, alpha=0.05):
    n1 = len(grupo1)
    n2 = len(grupo2)
    media_grupo1 = grupo1.mean()
    media_grupo2 = grupo2.mean()
    desvio_grupo1 = grupo1.std()
    desvio_grupo2 = grupo2.std()

    if desvio_grupo1 == 0:
        amostra_grupo1 = np.full(n1, media_grupo1)
    else:
        amostra_grupo1 = np.random.normal(media_grupo1, desvio_grupo1, n1)
    amostra_grupo2 = np.random.normal(media_grupo2, desvio_grupo2, n2)

    _, p = stats.wilcoxon(amostra_grupo1, amostra_grupo2, alternative='two-sided')
    return p < alpha

def calcular_poder_wilcoxon(grupo1, grupo2, alpha=0.05, num_simulacoes=1000):
    n1 = len(grupo1)
    n2 = len(grupo2)
    media_grupo1 = grupo1.mean()
    media_grupo2 = grupo2.mean()
    desvio_grupo1 = grupo1.std()
    desvio_grupo2 = grupo2.std()

    contagem_rejeicoes = 0
    for _ in range(num_simulacoes):
        # Gerar amostras simuladas (com desvio padrão 0, gera valores iguais)
        if desvio_grupo1 == 0:
          amostra_grupo1 = np.full(n1, media_grupo1)
        else:
          amostra_grupo1 = np.random.normal(media_grupo1, desvio_grupo1, n1)
        amostra_grupo2 = np.random.normal(media_grupo2, desvio_grupo2, n2)

        # Executar o teste de wilcoxon
        u, p = stats.wilcoxon(amostra_grupo1, amostra_grupo2, alternative='two-sided')

        # Verificar se a hipótese nula foi rejeitada
        if p < alpha:
            contagem_rejeicoes += 1

    poder = contagem_rejeicoes / num_simulacoes
    return poder


def calcular_mediana_diferenca_bootstrap(grupo1, grupo2, num_bootstraps=10000, nivel_confianca=0.99):
    diferencas = []
    for _ in range(num_bootstraps):
        amostra1 = random.choices(grupo1, k=len(grupo1))
        amostra2 = random.choices(grupo2, k=len(grupo2))
        diferencas_amostra = [x - y for x in amostra1 for y in amostra2]
        diferencas.append(np.median(diferencas_amostra))

    alpha = 1 - nivel_confianca
    limite_inferior = np.quantile(diferencas, alpha / 2)
    limite_superior = np.quantile(diferencas, 1 - alpha / 2)
    mediana_diferenca = np.median(diferencas)
    return mediana_diferenca, limite_inferior, limite_superior

def concordancia(cenarios, class_codes, classes, df_geral, modelos):
    for cenario in cenarios:
        print()
        print(f'Scores de {cenario}')
        print('\t', '\t'.join(modelos), end='')
        for classe in classes:
            print()
            print(f'{class_codes[classe]}\t', end='')
            tab = ''
            for modelo in modelos:
                if modelo.startswith('Llama'): tab = '\t'
                df_filtrado = df_geral[df_geral.index.str.contains(cenario + '-' + classe)]
                print(f'{df_filtrado.loc[df_filtrado[modelo] != 0, modelo].mean():.1f}\t\t' + tab, end='')


def analyze_agreement_kappa(transformed_dfs):
    """
    Calculates the agreement between each LLM and the 'SCO' column using Cohen's Kappa (κ).

    This function iterates through each scenario's DataFrame and computes the Kappa
    score, which measures inter-rater agreement while correcting for agreement
    that might occur by chance.

    Args:
        transformed_dfs (dict): A dictionary where keys are scenario names and
                                values are the transformed pandas DataFrames.
                                Each DataFrame must contain an 'SCO' column.

    Returns:
        dict: A dictionary where keys are scenario names and values are pandas
              Series containing the Cohen's Kappa score for each LLM.
    """
    kappa_results = {}
    # Define the set of all possible classification labels.
    # This is important for ensuring Kappa is calculated correctly even if
    # a rater doesn't use all possible labels in a given sample.
    labels = [-1, 0, 1]

    for scenario, df in transformed_dfs.items():
        if 'SCO' not in df.columns:
            print(f"Warning: 'SCO' column not found in scenario '{scenario}'. Skipping.")
            continue

        sco_truth = df['SCO']
        llm_columns = df.columns.drop('SCO')

        scenario_kappas = {}
        for llm in llm_columns:
            # Calculate Cohen's Kappa for the current LLM vs. the SCO column
            kappa_value = cohen_kappa_score(sco_truth, df[llm], labels=labels)
            scenario_kappas[llm] = kappa_value

        # Store the results for the scenario as a pandas Series
        kappa_results[scenario] = pd.Series(scenario_kappas)

    return kappa_results

def analyze_agreement(transformed_dfs):
    """
    Analyzes and calculates the percentage agreement between each LLM column and the 'SCO' column.

    This function iterates through each scenario in the input dictionary. For each
    scenario's DataFrame, it compares every LLM's classification against the 'SCO'
    classification on a row-by-row basis.

    Args:
        transformed_dfs (dict): A dictionary where keys are scenario names and
                                values are the transformed pandas DataFrames.
                                Each DataFrame must contain an 'SCO' column and
                                one or more LLM columns.

    Returns:
        dict: A dictionary where keys are scenario names and values are
              pandas Series containing the agreement percentage for each LLM.
    """
    agreement_results = {}

    for scenario, df in transformed_dfs.items():
        # Ensure the 'SCO' column exists to avoid errors
        if 'SCO' not in df.columns:
            print(f"Warning: 'SCO' column not found in scenario '{scenario}'. Skipping.")
            continue

        # Isolate the ground truth column ('SCO') and the LLM prediction columns
        sco_truth = df['SCO']
        llm_columns = df.columns.drop('SCO')

        # Calculate the number of exact matches for each LLM column against the SCO column
        # The result of the comparison (==) is a boolean DataFrame. sum() treats True as 1 and False as 0.
        agreement_counts = df[llm_columns].eq(sco_truth, axis=0).sum()

        # Calculate the percentage by dividing by the total number of classes (rows)
        total_classes = len(df)
        agreement_percentage = (agreement_counts / total_classes) * 100

        agreement_results[scenario] = agreement_percentage

    return agreement_results


def print_latex_tables(agreement_results):
    """
    Generates and prints LaTeX tables from the agreement analysis results.

    The format of the table is specifically structured with multi-row headers
    grouping the models by family.

    Args:
        agreement_results (dict): A dictionary from the `analyze_agreement` function,
                                  where keys are scenario names and values are
                                  pandas Series of agreement percentages.
    """
    # --- 1. Mappings for dynamic content generation ---

    # Map scenario keys to human-readable titles for the LaTeX captions
    scenario_to_title = {
        'original': 'Original Code',
        'no_comments': 'Comments Removed',
        'bad_names': 'Identifier Names Obfuscated',
        'clean_code': 'Clean Code (Refactored)'
    }

    # Map scenario keys to labels for cross-referencing in LaTeX
    scenario_to_label = {
        'original': 'tab:agreementOriginal',
        'no_comments': 'tab:agreementNoComments',
        'bad_names': 'tab:agreementBadNames',
        'clean_code': 'tab:agreementCleanCode'
    }

    # Define the structure and order of model families for the table headers
    model_structure = {
        'ChatGPT': ['GPT4o', 'GPT4o_mini'],
        'Gemini': ['Gemini20pro', 'Gemini20flash'],
        'Llama': ['Llama31_405b', 'Llama31_8b'],
        'Claude': ['Claude37_sonnet', 'Claude35_haiku'],
        'Deep': ['DeepSeek_V3']
    }

    # Flatten the list of models to define the column order
    model_order = [model for family in model_structure.values() for model in family]

    # Map internal model names to their specific text in the header rows
    model_display_map = {
        'GPT4o': {'line2': '4o', 'line3': ' '},
        'GPT4o_mini': {'line2': '4o', 'line3': 'mini'},
        'Gemini20pro': {'line2': '2.0', 'line3': 'Pro'},
        'Gemini20flash': {'line2': '2.0', 'line3': 'Flash'},
        'Llama31_405b': {'line2': '3.1', 'line3': '405B'},
        'Llama31_8b': {'line2': '3.1', 'line3': '8B'},
        'Claude37_sonnet': {'line2': '3.7', 'line3': 'Sonnet'},
        'Claude35_haiku': {'line2': '3.5', 'line3': 'Haiku'},
        # Handle the special formatting for DeepSeek from your example
        'DeepSeek_V3': {'line2': r'\textbf{Seek}', 'line3': 'V3'}
    }

    # --- 2. Iterate through scenarios and generate each table ---

    for scenario, percentages in agreement_results.items():
        title = scenario_to_title.get(scenario, scenario.replace('_', ' ').title())
        label = scenario_to_label.get(scenario, f'tab:agreement{scenario.title()}')

        # Build each part of the LaTeX string

        # Header for the table environment
        latex_lines = [
            r'\begin{table}[htbp]',
            r'\vspace{-5pt}',
            r'    \centering',
            r'    \scriptsize',
            fr'    \caption{{Agreement with Scalabrino Classifier - {title}}}',
            fr'    \begin{{tabular}}{{l{"r" * len(model_order)}}}',  # l for the first col, r for each model
            r'        \toprule',
        ]

        # Header Row 1: Model Families (e.g., \multicolumn{2}{c}{\textbf{ChatGPT}})
        header1_parts = []
        for family, models_in_family in model_structure.items():
            col_span = len(models_in_family)
            header1_parts.append(fr'& \multicolumn{{{col_span}}}{{c}}{{\textbf{{{family}}}}} ')
        latex_lines.append("".join(header1_parts) + r'\\')

        # Mid-rule
        latex_lines.append(fr'        \cmidrule{{2-{len(model_order) + 1}}}')

        # Header Row 2: Model Versions (e.g., 4o, 2.0, 3.1)
        header2_parts = [r'         ']
        for model_name in model_order:
            header2_parts.append(f"& {model_display_map[model_name]['line2']} ")
        latex_lines.append("".join(header2_parts) + r'\\')

        # Header Row 3: Model Sub-versions (e.g., mini, Pro, 405B)
        header3_parts = [r'                        ']  # Initial empty cell
        header3_parts.extend([f"& {model_display_map[model_name]['line3']} " for model_name in model_order])
        latex_lines.append("".join(header3_parts) + r'\\')

        # Main data row
        latex_lines.append(r'        \midrule')
        data_row_parts = [r'        Agreement ']
        for model_name in model_order:
            # Format percentage to integer if whole number, else keep decimals
            val = percentages[model_name]
            formatted_val = f'{val:g}'  # 'g' format removes trailing .0
            data_row_parts.append(fr'& {formatted_val}\% ')
        latex_lines.append("".join(data_row_parts) + r'\\')

        # Footer for the table environment
        latex_lines.extend([
            r'        \bottomrule',
            r'    \end{tabular}',
            fr'    \label{{{label}}}',
            r'    \vspace{-5pt}',
            r'\end{table}',
            '\n'  # Add a newline for separation between tables
        ])

        # Print the complete LaTeX block for the scenario
        print("\n".join(latex_lines))

def print_agreement_description(agreement_results):
    """Formats and prints the agreement analysis results."""
    for scenario, percentages in agreement_results.items():
        print(f"Scenario: {scenario}")
        print("=" * (len(scenario) + 10))
        print("Percentage Agreement with SCO classification:")
        for llm, percentage in percentages.items():
            print(f"- {llm:<18}: {percentage:6.2f}%")
        print("\n" + "-"*40 + "\n")


def print_metrics_description(prf_results):
    """
    Formats and prints a text-based summary of precision, recall, and F1-score results.

    This function is designed to work with the output of `analyze_precision_recall_f1`,
    which is a dictionary of pandas DataFrames.

    Args:
        prf_results (dict): A dictionary from `analyze_precision_recall_f1`.
    """
    for scenario, metrics_df in prf_results.items():
        print(f"Scenario: {scenario}")
        print("=" * (len(scenario) + 10))

        # metrics_df is a DataFrame with models as the index and PRF as columns
        # We use .iterrows() to loop through each model's metrics
        for llm, metrics in metrics_df.iterrows():
            # llm is the model name (e.g., 'GPT4o')
            # metrics is a Series containing 'precision', 'recall', 'f1-score' for that model
            p = metrics['precision']
            r = metrics['recall']
            f1 = metrics['f1-score']

            print(f"- {llm:<18}: "
                  f"Precision: {p:.2f}, "
                  f"Recall: {r:.2f}, "
                  f"F1-Score: {f1:.2f}")

        print("\n" + "-" * 40 + "\n")

def analyze_precision_recall_f1(transformed_dfs):
    """
    Calculates macro-averaged precision, recall, and F1-score for each LLM
    against the 'SCO' ground truth column.

    Args:
        transformed_dfs (dict): A dictionary where keys are scenario names and
                                values are the transformed pandas DataFrames.

    Returns:
        dict: A dictionary where keys are scenario names and values are pandas
              DataFrames. Each DataFrame is indexed by the LLM name and has
              columns for 'precision', 'recall', and 'f1-score'.
    """
    results = {}
    # Define the set of all possible classification labels for consistency.
    labels = [-1, 0, 1]

    for scenario, df in transformed_dfs.items():
        if 'SCO' not in df.columns:
            print(f"Warning: 'SCO' column not found in scenario '{scenario}'. Skipping.")
            continue

        sco_truth = df['SCO']
        llm_columns = df.columns.drop('SCO')

        scenario_metrics = []
        for llm in llm_columns:
            # Use scikit-learn to compute the metrics.
            # 'average="macro"' calculates metrics for each label, then finds their unweighted mean.
            # 'zero_division=0' prevents warnings and returns 0.0 if a metric is ill-defined.
            p, r, f1, _ = precision_recall_fscore_support(
                sco_truth,
                df[llm],
                average='weighted',
                labels=labels,
                zero_division=0
            )
            scenario_metrics.append({
                'model': llm,
                'precision': p,
                'recall': r,
                'f1-score': f1
            })

        # Convert the list of dictionaries to a DataFrame for easy handling
        results_df = pd.DataFrame(scenario_metrics).set_index('model')
        results[scenario] = results_df

    return results


def print_prf_latex_tables(prf_results):
    """
    Generates and prints LaTeX tables for precision, recall, and F1-score results.
    The table includes three data rows for the three metrics.

    Args:
        prf_results (dict): A dictionary from `analyze_precision_recall_f1`.
    """
    # --- Mappings are the same as before ---
    scenario_to_title = {
        'original': 'Original Code',
        'no_comments': 'Comments Removed',
        'bad_names': 'Identifier Names Obfuscated',
        'clean_code': 'Clean Code (Refactored)'
    }
    scenario_to_label = {
        'original': 'tab:prfOriginal',
        'no_comments': 'tab:prfNoComments',
        'bad_names': 'tab:prfBadNames',
        'clean_code': 'tab:prfCleanCode'
    }
    model_structure = {
        'ChatGPT': ['GPT4o', 'GPT4o_mini'],
        'Gemini': ['Gemini20pro', 'Gemini20flash'],
        'Llama': ['Llama31_405b', 'Llama31_8b'],
        'Claude': ['Claude37_sonnet', 'Claude35_haiku'],
        'Deep': ['DeepSeek_V3']
    }
    model_order = [model for family in model_structure.values() for model in family]
    model_display_map = {
        'GPT4o': {'line2': '4o', 'line3': ' '},
        'GPT4o_mini': {'line2': '4o', 'line3': 'mini'},
        'Gemini20pro': {'line2': '2.0', 'line3': 'Pro'},
        'Gemini20flash': {'line2': '2.0', 'line3': 'Flash'},
        'Llama31_405b': {'line2': '3.1', 'line3': '405B'},
        'Llama31_8b': {'line2': '3.1', 'line3': '8B'},
        'Claude37_sonnet': {'line2': '3.7', 'line3': 'Sonnet'},
        'Claude35_haiku': {'line2': '3.5', 'line3': 'Haiku'},
        'DeepSeek_V3': {'line2': r'\textbf{Seek}', 'line3': 'V3'}
    }

    # --- Iterate through scenarios and generate each table ---
    for scenario, results_df in prf_results.items():
        title = scenario_to_title.get(scenario, scenario.replace('_', ' ').title())
        label = scenario_to_label.get(scenario, f'tab:prf{scenario.title()}')

        # --- Build LaTeX Header (same as before) ---
        latex_lines = [
            r'\begin{table}[htbp]',
            r'\vspace{-5pt}',
            r'    \centering',
            r'    \scriptsize',
            fr'    \caption{{Precision, Recall, and F1-score vs. Scalabrino Classifier - {title}}}',
            fr'    \begin{{tabular}}{{l{"r" * len(model_order)}}}',
            r'        \toprule',
        ]
        header1_parts = []
        for family, models in model_structure.items():
            header1_parts.append(fr'& \multicolumn{{{len(models)}}}{{c}}{{\textbf{{{family}}}}} ')
        latex_lines.append("".join(header1_parts) + r'\\')
        latex_lines.append(fr'        \cmidrule{{2-{len(model_order) + 1}}}')
        header2_parts = [r'        \textbf{Metric} ']
        header2_parts.extend([f"& {model_display_map[name]['line2']} " for name in model_order])
        latex_lines.append("".join(header2_parts) + r'\\')
        header3_parts = [r'                  & ']
        header3_parts.extend([f"& {model_display_map[name]['line3']} " for name in model_order])
        latex_lines.append("".join(header3_parts) + r'\\')
        latex_lines.append(r'        \midrule')

        # --- Build LaTeX Data Rows (New Logic) ---
        for metric in ['precision', 'recall', 'f1-score']:
            metric_label = metric.replace('_', '-').title()
            row_parts = [f"        {metric_label} "]
            for model_name in model_order:
                # Get the value from the results DataFrame
                val = results_df.loc[model_name, metric]
                row_parts.append(f'& {val*100:.2f}\% ')
            latex_lines.append("".join(row_parts) + r'\\')

        # --- Build LaTeX Footer (same as before) ---
        latex_lines.extend([
            r'        \bottomrule',
            r'    \end{tabular}',
            fr'    \label{{{label}}}',
            r'    \vspace{-5pt}',
            r'\end{table}',
            '\n'
        ])
        print("\n".join(latex_lines))

def apply_classification_rules(dfs_with_sco):
    """
    Applies a set of classification rules to every numerical value in each DataFrame
    within a dictionary of DataFrames.

    The rules are:
    - Values from 60 to 100 (inclusive) are mapped to 1.
    - Values between 41.6 and 60 (exclusive) are mapped to -1.
    - Values from 0 to 41.6 (inclusive) are mapped to 0.

    Args:
        dfs_with_sco (dict): A dictionary where keys are scenario names and values
                             are pandas DataFrames containing numerical scores.

    Returns:
        dict: A new dictionary of DataFrames with the transformation rules applied.
              The original dictionary is not modified.
    """
    # Create a deep copy to ensure the original data structure is not mutated.
    transformed_dfs = {scenario: df.copy() for scenario, df in dfs_with_sco.items()}

    def classify_value(value):
        """Helper function to apply the classification logic to a single value."""
        if not pd.isna(value):
            if 60 <= value <= 100:
                return 1
            elif 41.6 < value < 60:
                return -1
            elif 0 <= value <= 41.6:
                return 0
        return value # Return the original value if it's NaN or doesn't fit the criteria

    # Iterate through the copied dictionary and apply the function to each DataFrame
    for scenario, df in transformed_dfs.items():
        transformed_dfs[scenario] = df.applymap(classify_value)

    return transformed_dfs

def add_sco_column(dfs, classifier_data, class_codes):
    """
    Adds a 'SCO' column to each DataFrame in a dictionary of DataFrames.

    The function maps scores from the classifier_data DataFrame to the
    correct scenario and class in the dfs dictionary.

    Args:
        dfs (dict): A dictionary of pandas DataFrames, with scenario names as keys.
                    Each DataFrame is indexed by class codes.
        classifier_data (pd.DataFrame): A DataFrame with a 'sco' column,
                                        indexed by a string like 'scenario-classname'.
        class_codes (dict): A dictionary mapping full class names (e.g., 'Month.java')
                            to class codes (e.g., 'C1', 'C2').

    Returns:
        dict: A new dictionary of DataFrames, where each DataFrame includes the added 'SCO' column.
              The original dictionary is not modified.
    """
    # Create a deep copy to avoid modifying the original dictionary of DataFrames
    updated_dfs = {scenario: df.copy() for scenario, df in dfs.items()}

    # For mapping class names back to their codes efficiently
    # Example: {'DoubleSummaryStatistics.java': 'C1', ...}
    # This is the provided class_codes dictionary

    for scenario, df in updated_dfs.items():
        # Filter classifier_data for the current scenario
        # This selects rows where the index starts with the scenario name, e.g., 'original-'
        scenario_scores = classifier_data[classifier_data.index.str.startswith(f'{scenario}-')]

        # Extract the class name from the index (e.g., 'DoubleSummaryStatistics.java')
        # and map it to the corresponding class code (e.g., 'C1')
        def get_class_code(index_label):
            class_name = index_label.replace(f'{scenario}-', '')
            return class_codes.get(class_name)

        # Create a Series of scores indexed by the class code ('C1', 'C2', etc.)
        sco_series = scenario_scores['sco'].rename(index=get_class_code)

        # Add the 'SCO' column to the DataFrame by mapping the scores using the index
        df['SCO'] = df.index.map(sco_series)

        # Round the new column to a desired precision, e.g., 2 decimal places
        df['SCO'] = df['SCO'].round(2)

    return updated_dfs

def concordancia_df(cenarios, class_codes, classes, df_geral, modelos):
    results = {}

    for cenario in cenarios:
        # List to hold the data for the current scenario's DataFrame
        data_for_df = []

        for classe in classes:
            # Dictionary to build a single row of the DataFrame
            row_data = {'Class': class_codes[classe]}

            for modelo in modelos:
                # Filter the DataFrame for the specific scenario and class
                df_filtrado = df_geral[df_geral.index.str.contains(f'{cenario}-{classe}')]

                # Calculate the mean for non-zero values
                # Using .get(modelo, pd.NA) handles cases where a model column might be missing for a slice
                mean_value = df_filtrado.loc[df_filtrado.get(modelo, 0) != 0, modelo].mean()

                # Store the calculated mean in our row dictionary
                row_data[modelo] = mean_value

            # Add the completed row to our list of data
            data_for_df.append(row_data)

        # Create the DataFrame for the current scenario
        df_scenario = pd.DataFrame(data_for_df)
        df_scenario.set_index('Class', inplace=True)

        # Store the DataFrame in our results dictionary
        results[cenario] = df_scenario.round(1)  # Rounding to one decimal place as in the print statement

    # If only one scenario was processed, return the DataFrame directly.
    # Otherwise, return the dictionary of DataFrames.
    if len(cenarios) == 1:
        return results[cenarios[0]]

    return results



def tabela_mean(cenarios, class_codes, classes, df_geral, modelos):
    for cenario in cenarios:
        print()
        print(f'Scores de {cenario}')
        print('\t', '\t'.join(modelos), end='')
        for classe in classes:
            print()
            print(f'{class_codes[classe]}\t', end='')
            tab = ''
            for modelo in modelos:
                if modelo.startswith('Llama'): tab = '\t'
                df_filtrado = df_geral[df_geral.index.str.contains(cenario + '-' + classe)]
                print(f'{df_filtrado.loc[df_filtrado[modelo] != 0, modelo].mean():.1f}\t\t' + tab, end='')
    for cenario in cenarios:
        if cenario != 'original':
            print()
            print(f'Scores de {cenario}')
            print('\t', '\t\t'.join(modelos), end='')
            for classe in classes:
                print()
                print(f'{class_codes[classe]}\t', end='')
                tab = ''
                for modelo in modelos:
                    if modelo.startswith('Llama'): tab = '\t'
                    df_original = df_geral[df_geral.index.str.contains('original' + '-' + classe)]
                    df_filtrado = df_geral[df_geral.index.str.contains(cenario + '-' + classe)]
                    value = -100 * (1 - (df_filtrado.loc[df_filtrado[modelo] != 0, modelo].mean())/(df_original.loc[df_original[modelo] != 0, modelo].mean()))
                    print(f'{value:.1f}{"\t\t\t\t" + tab if value >= 0.1 else "\t\t\t" + tab}', end='')

def tabela_std(cenarios, class_codes, classes, df_geral, modelos):
        for cenario in cenarios:
            print()
            print(f'Desvio padrão e Coeficiente de Variação de {cenario}')
            print('\t', '\t\t\t'.join(modelos), end='')
            for classe in classes:
                print()
                print(f'{class_codes[classe]}\t', end='')
                tab = ''
                for modelo in modelos:
                    if modelo.startswith('Llama'):
                        tab = '\t'
                    df_filtrado = df_geral[df_geral.index.str.contains(cenario + '-' + classe)]
                    valores_filtrados = df_filtrado.loc[df_filtrado[modelo] != 0, modelo]
                    std = valores_filtrados.std()
                    media = valores_filtrados.mean()
                    cv = (std / media) * 100 if media != 0 else np.nan  # Calcula o CV, lidando com divisão por zero
                    if std == 0: print('-\t\t\t\t\t' + tab, end='')
                    else: print(f'{std:.1f} ({cv:.2f}%)\t\t\t' + tab, end='')


def calcular_variancia_df(df):
    # Itera sobre as colunas
    for nome_coluna in df.columns:
        coluna = df[nome_coluna]

        # Verifica se a coluna é numérica
        if pd.api.types.is_numeric_dtype(coluna):
            # Calcula a variância, tratando NA's
            variancia = coluna.var(skipna=True)  # skipna=True ignora valores NA
            # Imprime o resultado formatado
            print(f"Variância da coluna '{nome_coluna}': {variancia}")
            print(coluna.describe())
        else:
            print(f"A coluna '{nome_coluna}' não é numérica. Cálculo da variância ignorado.")

def plotar_scatter(df, coluna_x, coluna_y):
    plt.figure(figsize=(8, 6))  # Define o tamanho da figura
    plt.scatter(df[coluna_x], df[coluna_y], alpha=0.7)  # alpha controla a transparência dos pontos

    plt.title(f'Gráfico de Dispersão: {coluna_x} vs {coluna_y}')
    plt.xlabel(coluna_x)
    plt.ylabel(coluna_y)
    plt.grid(True)  # Adiciona um grid ao gráfico (opcional)
    plt.tight_layout() # Ajusta o layout para evitar sobreposição de elementos
    plt.show()

def plotar_histogramas(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    num_cols = len(numeric_cols)

    if num_cols == 0:
        print("Não há colunas numéricas no DataFrame.")
        return

    for col in numeric_cols:
        plt.figure(figsize=(8, 6))  # Cria uma nova figura para cada histograma
        plt.hist(df[col], bins=10, edgecolor='black')
        plt.title(f'Distribuição de {col}')
        plt.xlabel(col)
        plt.ylabel('Frequência')
        plt.tight_layout() # Ajusta o layout para evitar sobreposição
        plt.show()



def criar_data_frame_geral(data_lists):
    col_names = ['name', 'score', 'code_smells', 'cognitive_complexity', 'comment_lines_density', 'complexity', 'lines',
                 'sqale_rating', 'statements']
    dfs = []
    for key in data_lists.keys():
        if all(isinstance(row, list) for row in data_lists[key]):
            df = pd.DataFrame(data_lists[key], columns=col_names)
            df.rename(columns={'score':key.split("-")[0]}, inplace=True)
            df['name'] = key.split("-")[1] + '-' + df['name']
            dfs.append(df)

    df_final = criar_matriz_de_dataframes(dfs)
    # print(df_final.to_string())
    # df_final['Claude35_sonnet'] = df_final['Claude35_sonnet'].astype(int)
    # df_final['Claude3_haiku'] = df_final['Claude3_haiku'].astype(int)
    # df_final['GPT4o'] = df_final['GPT4o'].astype(int)
    # df_final['GPT4o_mini'] = df_final['GPT4o_mini'].astype(int)
    df_final['Gemini15flash'] = df_final['Gemini15flash'].astype(int)
    # df_final['Gemini15pro'] = df_final['Gemini15pro'].astype(int)
    # df_final['Llama31_405b'] = df_final['Llama31_405b'].astype(int)
    df_final['Llama31_8b'] = df_final['Llama31_8b'].astype(int)
    df_final['code_smells'] = df_final['code_smells'].astype(float)
    df_final['cognitive_complexity'] = df_final['cognitive_complexity'].astype(float)
    df_final['comment_lines_density'] = df_final['comment_lines_density'].astype(float)
    df_final['complexity'] = df_final['complexity'].astype(float)
    df_final['lines'] = df_final['lines'].astype(float)
    df_final['sqale_rating'] = df_final['sqale_rating'].astype(float)
    df_final['statements'] = df_final['statements'].astype(float)
    return df_final

def merge_dataframes_customizado(dfs, chave='name'):
    if not dfs:
        return None

    df_merged = dfs[0].set_index(chave).copy() #Define o index no primeiro dataframe e o copia

    for i, df in enumerate(dfs[1:]):
        try:
            df = df.set_index(chave)
        except KeyError:
            print(f"Erro: A coluna '{chave}' não existe no DataFrame {i + 2}.")
            return None

        for coluna in df.columns:
            if coluna in df_merged.columns:
                print(f"Conflito na coluna '{coluna}' entre DataFrame inicial e DataFrame {i + 2}:")
                for indice in df.index:
                    if indice in df_merged.index:
                        valor_df = df.at[indice, coluna]
                        valor_merged = df_merged.at[indice, coluna]
                        print(f'valor_df: {valor_df}')
                        if not pd.isna(valor_df) and not pd.isna(valor_merged):
                            print( f"  Índice: '{indice}' - Valor no DataFrame inicial: '{valor_merged}', Valor no DataFrame {i + 2}: '{valor_df}'")
                            df_merged.at[indice, coluna] = valor_df
                        elif not pd.isna(valor_df) and pd.isna(valor_merged):
                            df_merged.loc[indice, coluna] = valor_df
            else:
                df_merged = df_merged.join(df[coluna], how='outer')

    return df_merged

def tornar_indices_unicos(df):
    indices_originais = df.index
    novos_indices = []
    contador = {}

    for indice in indices_originais:
        if indice in contador:
            contador[indice] += 1
            novos_indices.append(f"{contador[indice]}-{indice}")
        else:
            contador[indice] = 1
            novos_indices.append(f"{contador[indice]}-{indice}")

    df.index = novos_indices
    return df

def criar_matriz_de_dataframes(dfs, chave='name'):
    if not dfs:
        print("Erro: A lista de DataFrames está vazia.")
        return None

    if not all(isinstance(df, pd.DataFrame) for df in dfs):
        print("Erro: Todos os elementos em 'dfs' devem ser DataFrames.")
        return None

    indices_unicos = set()
    colunas_unicas = set()

    for df in dfs:
        try:
            df = df.set_index(chave)
            df = tornar_indices_unicos(df)
            df.reset_index(drop=True)
        except KeyError:
            print(f"Erro: A coluna '{chave}' não existe em um dos DataFrames.")
            return None
        indices_unicos.update(df.index)
        colunas_unicas.update(df.columns)

    indices_unicos = sorted(list(indices_unicos))
    colunas_unicas = sorted(list(colunas_unicas))

    # print('Indices:', indices_unicos)
    print('Colunas:', colunas_unicas)

    df_final = pd.DataFrame(index=indices_unicos, columns=colunas_unicas)
    for df in dfs:
        df = df.set_index('name')
        df = tornar_indices_unicos(df)
        # print(df.to_string())
        for col in df.columns:
            if col in df_final.columns:
                df_final.loc[df.index, col] = df.loc[df.index, col]
    return df_final

def matriz_correlacao_geral(df):
    # Removendo a coluna 'name' para a análise de correlação (é categórica)
    df_numeric = df.copy()

    # Calculando a matriz de correlação
    correlation_matrix = df_numeric.corr()

    # Criando o heatmap
    plt.figure(figsize=(10, 8))  # Ajusta o tamanho da figura para melhor visualização
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")  # adicionado fmt para formatar os numeros
    plt.title(f'Matriz de Correlação')
    plt.show()

    print("Matriz de Correlação:\n", correlation_matrix)

def matriz_correlacao(cenario, data_lists):
    # Nomes das colunas
    col_names = ['name', 'score', 'code_smells', 'cognitive_complexity', 'comment_lines_density', 'complexity', 'lines',
                 'sqale_rating', 'statements']

    # Criando o DataFrame
    df = pd.DataFrame(data_lists, columns=col_names)
    # Removendo a coluna 'name' para a análise de correlação (é categórica)
    df_numeric = df.drop('name', axis=1)

    # Calculando a matriz de correlação
    correlation_matrix = df_numeric.corr()

    # Criando o heatmap
    plt.figure(figsize=(10, 8))  # Ajusta o tamanho da figura para melhor visualização
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")  # adicionado fmt para formatar os numeros
    plt.title(f'Matriz de Correlação no cenario {cenario}')
    plt.show()

    print("Matriz de Correlação:\n", correlation_matrix)


def clusterizar(scores):

    for key, values in scores.items():
        mean_value = int(np.mean([v for v in values if v != 0]))
        print(f'Média encontrada: {mean_value} para {key}-{values}')
        scores[key] = [mean_value if v == 0 else v for v in values]

    # Convertendo os dados para um formato adequado para o K-Means
    data = np.array(list(scores.values()))

    # Aplicando K-Means
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
    print('clusterizado', kmeans)

    # Agrupando os labels por cluster
    clusters = {}
    for key, label in zip(scores.keys(), kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(key)

    # Exibindo os rótulos dos clusters
    for label, keys in clusters.items():
        print(f"Cluster {label}: {', '.join(keys)}")

def teste_geral_mannwhitneyu(data_lists):
    scores = {}
    for key in data_lists.keys():
        if all(isinstance(row, list) for row in data_lists[key]):
            scores[key] = [row[1] for row in data_lists[key]]# if row[1] != 0
    for key in scores.keys():
        print(key, scores[key])
    realizar_teste_mannwhitneyu(scores)
    clusterizar(scores)

def realizar_teste_mannwhitneyu(scores, alpha=0.05):
    if len(scores) < 2:
        print("São necessários pelo menos dois grupos para realizar o teste de Mann-Whitney U.")
        return {}

    keys = list(scores.keys())
    resultados_agrupados = defaultdict(list)
    resultados_ordenados = []

    for grupo1, grupo2 in itertools.combinations(keys, 2):
        valores_grupo1 = scores[grupo1]
        valores_grupo2 = scores[grupo2]
        n1 = len(valores_grupo1)
        n2 = len(valores_grupo2)
        n_total = n1 + n2

        try:
            stat, p_value = mannwhitneyu(valores_grupo1, valores_grupo2)
        except ValueError as e:
            print(f"Erro ao comparar {grupo1} e {grupo2}: {e}")
            print("Verifique se os grupos têm dados suficientes e se os dados são numéricos.")
            return {}

        resultados_ordenados.append({
            "grupo1": grupo1,
            "grupo2": grupo2,
            "stat": stat,
            "p_value": p_value,
            "n_total": n_total
        })

    p_valores = [item['p_value'] for item in resultados_ordenados]
    reject, pvals_corrigidos, _, _ = multipletests(p_valores, alpha=alpha, method='bonferroni')

    for i, item in enumerate(resultados_ordenados):
        z = norm.ppf(1 - item['p_value'] / 2)
        r = z / np.sqrt(item['n_total'])
        p_value_corrigido = pvals_corrigidos[i]

        if p_value_corrigido < 0.001:
            grupo_significancia = "*** (p < 0.001)"
        elif p_value_corrigido < 0.01:
            grupo_significancia = "** (p < 0.01)"
        elif p_value_corrigido < 0.05:
            grupo_significancia = "* (p < 0.05)"
        else:
            grupo_significancia = "Não Significativo (p >= 0.05)"

        resultados_ordenados[i]["r"] = r
        resultados_ordenados[i]["p_value_corrigido"] = p_value_corrigido
        resultados_ordenados[i]["grupo_significancia"] = grupo_significancia

    # Agrupa os resultados pelo grupo de significância
    for item in resultados_ordenados:
        resultados_agrupados[item["grupo_significancia"]].append(item)

    # Ordena os resultados DENTRO DE CADA GRUPO de significância
    for grupo_significancia in resultados_agrupados:
        resultados_agrupados[grupo_significancia].sort(key=lambda x: abs(x["r"]),
                                                       reverse=True)  # ordena pelo modulo de r

    print(
        f"Resultados com correção de Bonferroni (alpha = {alpha}), agrupados por significância e ordenados por |r| (decrescente) dentro de cada grupo:")

    for significancia, resultados_dentro_do_grupo in resultados_agrupados.items():
        print(f"\nComparações com significância {significancia}:")
        for item in resultados_dentro_do_grupo:
            print(
                f"  - {item['grupo1']} vs {item['grupo2']} (U={item['stat']:.2f}, p={item['p_value_corrigido']:.3e}, r={item['r']:.2f})")

    return resultados_agrupados

def realizar_teste_levene(scores):
    levene_results = {}

    # Verificar se há pelo menos dois grupos para realizar o teste de Levene
    if len(scores) < 2:
      print("São necessários pelo menos dois grupos para realizar o teste de Levene.")
      return levene_results

    # Extrair os valores para o teste de Levene
    values_list = list(scores.values())

    # Teste de Levene original
    stat, p_value = levene(*values_list) # O * desempacota a lista de valores
    levene_results["original"] = (stat, p_value)

    print(f'Resultados do teste de Levene (original):')
    print(f'Estatística: {stat}, Valor p: {p_value}\n')

    return levene_results

def realizar_teste_shapiro(scores):
    for key, values in scores.items():
        # Teste original
        stat, p_value = shapiro(values)

        print(f'Resultados do teste de Shapiro-Wilk para {key} (original):')
        print(f'Estatística: {stat}, Valor p: {p_value}\n')

        # Transformação logaritmica
        transformed_values = np.log(values)
        stat_transformed, p_value_transformed = shapiro(transformed_values)

        print(f'Resultados do teste de Shapiro-Wilk para {key} (transformação log):')
        print(f'Estatística: {stat_transformed}, Valor p: {p_value_transformed}\n')

def gerar_boxplot(data_lists, llm_lists, data_list_name):
    scores = {}
    for llm_list in llm_lists.keys():
        key = llm_list + '-' + data_list_name
        scores[key] = [row[1] for row in data_lists[key] if row[1] != 0]

    # Realizar o teste de Shapiro-Wilk
    # realizar_teste_shapiro(scores)
    # realizar_teste_levene(scores)
    # realizar_teste_mannwhitneyu(scores)

    # Gerar o gráfico de boxplot
    fig, ax = plt.subplots()
    ax.boxplot(scores.values())
    ax.set_ylim([0, 100])  # Ajuste da escala do eixo Y aqui
    ax.set_xticklabels(llm_lists.values(), rotation=20, ha='right')

    # plt.title(f'Boxplot dos Scores para {data_list_name}')
    plt.tight_layout()
    plt.savefig(f'graficos/boxplot_all_{data_list_name}.png', dpi=300)  # Salvar com alta resolução

def gerar_boxplot_sobreposto(data_lists, llm_lists, data_list_name, data_list_name2):
    label_lists = {
        'clean_code': 'I3',
        # 'bad_names_no_comments': 'I3',
        'bad_names': 'I2',
        'no_comments': 'I1',
        'original': 'OC'
    }
    scores = {}
    for llm_list in llm_lists.keys():
        key1 = llm_list + '-' + data_list_name
        key2 = llm_list + '-' + data_list_name2
        scores[key1] = [row[1] for row in data_lists[key1] if row[1] != 0]
        scores[key2] = [row[1] for row in data_lists[key2] if row[1] != 0]

    # Gerar o gráfico de boxplot
    fig, ax = plt.subplots()
    boxplot_data = []
    labels = []
    group_labels = []
    for llm_list in llm_lists.keys():
        key1 = llm_list + '-' + data_list_name
        key2 = llm_list + '-' + data_list_name2
        boxplot_data.append(scores[key1])
        boxplot_data.append(scores[key2])
        labels.append(label_lists[data_list_name])
        labels.append(label_lists[data_list_name2])
        group_labels.append(llm_list)

    ax.boxplot(boxplot_data)
    ax.set_ylim([0, 100])  # Ajuste da escala do eixo Y aqui

    # Adicionar linhas verticais a cada 2 boxplots
    for i in range(2, len(labels), 2):
        ax.axvline(x=i + 0.5, color='grey', linestyle='--')

    # Adicionar labels de grupo abaixo do gráfico
    ax.set_xticks([i + 1.5 for i in range(0, len(labels), 2)])
    ax.set_xticklabels(group_labels, rotation=35, ha='right', fontsize=10)

    # Adicionar rótulos adicionais na parte superior do gráfico
    for i in range(len(labels)):
        ax.text(i + 1, 105, labels[i], ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()
    plt.savefig(f'graficos/bp_{data_list_name}_vs_{data_list_name2}.png', dpi=300)  # Salvar com alta resolução



def loadData(data, file):
    if not os.path.exists(file):
        print(f"Erro: O arquivo '{file}' não existe.")
        return
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


def calcular_variacao(df):
    # Suponha que 'df' é o seu DataFrame e que 'name' e 'score' são colunas no DataFrame
    df['score'] = df['score'].astype(int)  # Certifique-se de que a coluna 'score' é de tipo int
    # Agrupe por 'name' e calcule a variação (max - min) para 'score'
    print('Média dos Scores:')
    print(df.groupby('name')['score'].mean())
    score_variation = df.groupby('name')['score'].apply(lambda x: x.max() - x.min())
    print('Amplitude de variação:')
    print(score_variation)
    # Para cada 'name', calcule o valor-p do Teste de Wilcoxon
    for name, group in df.groupby('name'):
        scores = group['score'].values
        if np.std(scores) != 0:  # Se não houver variação nos scores, o teste não pode ser aplicado
            w, p = wilcoxon(scores - np.mean(scores))
            print(f"O valor-p para o Teste de Wilcoxon para {name} é {p} e w={w}")
        # Para cada 'name', calcule o valor-p do Teste de Sinais
    for name, group in df.groupby('name'):
        scores = group['score'].values
        if len(scores) > 1:
            median = np.median(scores)
            differences = scores - median
            num_positives = np.sum(differences > 0)
            num_negatives = np.sum(differences < 0)
            if num_positives + num_negatives >= 1:
                # Use a distribuição binomial para calcular o valor-p
                p = binomtest(min(num_positives, num_negatives), num_positives + num_negatives, 0.5)
                print(f"O valor-p para o Teste de Sinais para {name} é {p}.")
    # Para cada 'name', calcule o percentual de registros com valores diferentes
    comulative_percentage = 0
    for name, group in df.groupby('name'):
        scores = group['score'].values
        unique_scores = np.unique(scores)
        if len(unique_scores) == 1:  # Se todos os scores são iguais
            percentage = 0
        else:
            percentage = (len(unique_scores) - 1) / len(scores) * 100
        print(f"{percentage}% de registros com valores diferentes para {name}")
        comulative_percentage += percentage
    print(f"{comulative_percentage / 1200 * 100}% de variação total")


def calculate_outliers(df, column):
    if len(df) == 0:
        return
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers) / len(df)


def calcular_estatisticas(df):
    if len(df) == 0:
        return
    # Filtrar as linhas onde 'score' é zero
    df = df[df['score'] != 0]

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



#                                             Claude35_haiku Claude35_sonnet Claude37_sonnet DeepSeek_V3 GPT4o GPT4o_mini  Gemini15flash Gemini15pro
# 1-bad_names-DoubleSummaryStatistics.java                85              35              20          40    65         65             40          10
# 1-bad_names-DynamicTreeNode.java                        75              45              35          75    75         75             40          10
# 1-bad_names-ElementTreePanel.java                       75              45              40          65    75         75             40          10
# 1-bad_names-HelloWorld.java                             65              45              30          40    75         65             20          10
# 1-bad_names-Month.java                                  95              95              65          85    85         85             75          10
# 1-bad_names-Notepad.java                                65              45              30          45    65         65             20          10
# 1-bad_names-SampleData.java                             65              45              35          65    75         70             20          10
# 1-bad_names-SampleTree.java                             85              45              35          65    85         75             40          10
# 1-bad_names-SampleTreeCellRenderer.java                 65              45              30          75    85         75             30          10
# 1-bad_names-SampleTreeModel.java                        75              65              60          75    85         85             85          30
# 1-bad_names-Stylepad.java                               65              35              30          65    65         75             30          10
# 1-bad_names-Wonderland.java                             65              45              30          40    75         65             40          10
# 1-clean_code-DoubleSummaryStatistics.java               95              95              95          95    95         85             95          90
# 1-clean_code-DynamicTreeNode.java                       85              85              85          85    85         85             75          60
