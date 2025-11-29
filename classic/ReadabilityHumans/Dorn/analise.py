from pingouin import intraclass_corr
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import numpy as np
from scipy.stats import spearmanr
import random
import matplotlib.colors as mcolors

def loadHumanScores(language, sinippet_init):
    df = pd.read_csv(f'{language}.csv', header=None)
    snippet_columns = [f'snippets/{language}/{i}.jsnp' for i in range(sinippet_init, len(df.columns) + sinippet_init - 1)]
    df.columns = ['human_id'] + snippet_columns
    return df

def loadHumanExperience():
    df = pd.read_csv('experience.csv', header=None)
    df.columns = ['human_id', 'Overall', 'Java', 'Cuda', 'Python', 'School', 'Industry']
    return df

def merge_dataframes_by_human_id(df1, df2):
    try:
        df1['human_id'] = df1['human_id'].astype('int64')
        df2['human_id'] = df2['human_id'].astype('int64')
        merged_df = pd.merge(df1, df2, on='human_id', how='inner')
        return merged_df
    except KeyError:
        print("Erro: A coluna 'human_id' não existe em um dos DataFrames.")
        return None
    except Exception as e:  # Captura outras possíveis exceções
        print(f"Ocorreu um erro durante o merge: {e}")
        return None

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
  df[column] = df[column].astype(int)/20
  return df.copy()

def merge_dataframes(df1, df2, index_name):
    df2.index.name = index_name
    merged_df = df1.join(df2)
    return merged_df

def loadAndMerge(scenario, llm, df_param):
    all_files = os.listdir()
    files = [file for file in all_files if file.startswith(scenario+llm)]
    counter = 1
    for file in files:
        loaded_df = load_json_data(file, f'{llm}-{counter}')
        df_param = merge_dataframes(df_param.copy(), loaded_df.copy(), f'{llm}-{counter}')
        counter = counter+1
    return df_param

def load(scenario, llm):
    all_files = os.listdir()
    files = [file for file in all_files if file.startswith(scenario+llm)]
    counter = 1
    temp = pd.DataFrame()
    for file in files:
        loaded_df = load_json_data(file, f'{llm}-{counter}')
        temp[f'{llm}-{counter}'] = loaded_df[f'{llm}-{counter}']
        counter = counter+1
    return temp

def plot_boxplots(df):
    df_melted = df.melt(var_name='column', value_name='value')

    # Create the subplots and boxplots
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="column",
        y="value",
        showmeans=True,  # Include means as markers
        data=df_melted
    )
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    plt.title('Box Plots for All Columns')
    plt.savefig(f'graficos/boxplots_TODOS.png', dpi=300)

def plot_violins(df, file_name):
    df_melted = df.melt(var_name='Column', value_name='Value')

    plt.figure(figsize=(12, 9))
    sns.violinplot(x='Column', y='Value', data=df_melted, orient='v')
    plt.title('Violin Plots for All Columns')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.savefig(f'graficos/violinos_{file_name}.png', dpi=300)
    plt.close()

def analyze_scores(df, column_one, column_two):
    W_stat, p_value = wilcoxon(df[column_one], df[column_two])
    return W_stat, p_value

def plot_pairplot(df, column_one, column_two):
    sns.pairplot(df[[column_one, column_two]], hue=column_one)
    plt.savefig(f'graficos/pairplot-{column_one}-{column_two}.png', dpi=300)

def plot_correlacao(df, column_one, column_two, llm):
    fig, ax = plt.subplots(figsize=(12, 8))  # Ajuste o tamanho aqui (largura, altura)

    sns.regplot(x=column_one, y=column_two, data=df, ax=ax, scatter_kws={'alpha':0.7}) # Usar regplot e ax

    ax.set_xlabel(column_one)  # Configurar os labels nos eixos
    ax.set_ylabel(column_two)
    ax.set_title(f'{column_one}({llm}) Vs {column_two}')

    plt.tight_layout()  # Ajusta o layout para evitar cortes
    plt.savefig(f'graficos/correlacao-{llm}-{column_one}-{column_two}.png', dpi=300)
    plt.close(fig)  # Fecha a figura para liberar memória


def plot_ccorelacao_multiplos(dfs, columns, file_name):
    num_plots = len(dfs)
    num_cols = 2
    num_rows = (num_plots + 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 6 * num_rows))
    axes = axes.flatten()

    for ax, (df, (column_one, column_two)) in zip(axes, zip(dfs, columns)):
        if not df[column_one].empty:
            sns.scatterplot(x=column_one, y=column_two, data=df, ax=ax)
            sns.regplot(x=column_one, y=column_two, data=df, scatter=False, ax=ax, color='blue')

            # Calculando a inclinação da linha de tendência
            x = df[column_one]
            y = df[column_two]
            slope, intercept = np.polyfit(x, y, 1)
            angle = np.degrees(np.arctan(slope))

            ax.set_xlabel(column_one)
            ax.set_ylabel(f'{column_two} size: {len(df)}')
            ax.set_title(f'{column_one} Vs {column_two} - angle: {angle:.2f}°')

    # Remover subgráficos vazios
    for ax in axes[num_plots:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(f'graficos/correlacao_{file_name}.png', dpi=300)
    plt.close()

def plot_scatter(df, column_one, column_two, column_three):
    sns.scatterplot(x=column_one, y=column_two, hue=column_three, data=df)
    sns.regplot(x=column_one, y=column_two, data=df, scatter=False, color='blue')
    plt.xlabel(column_one)
    plt.ylabel(column_two)
    plt.title(f'{column_one} Vs {column_two} com {column_three}')
    plt.savefig(f'graficos/scatter-{column_one}-{column_two}-{column_three}.png', dpi=300)

def calcular_medianas(df):
    try:
        medianas = {}
        if 'Valores' in df.columns:
            medianas['mediana_valores'] = df['Valores'].median()
            medianas['mediana_valores_sem_ignorar_nan'] = df['Valores'].median(skipna=False)

        medianas['medianas_todas'] = df.median(numeric_only=True)

        return medianas
    except Exception as e:
        print(f"Ocorreu um erro ao calcular as medianas: {e}")
        return None

def calcular_medias(df):
    try:
        medias = {}
        # Média de uma coluna específica
        if 'Idade' in df.columns:
            medias['media_idade'] = df['Idade'].mean()
        if 'Pontuação' in df.columns:
            medias['media_pontuacao'] = df['Pontuação'].mean()
            medias['media_pontuacao_sem_ignorar_nan'] = df['Pontuação'].mean(skipna=False)

        # Média de todas as colunas numéricas
        medias['medias_todas'] = df.mean(numeric_only=True)

        return medias
    except Exception as e:
        print(f"Ocorreu um erro ao calcular as médias: {e}")
        return None

def filter(df, category, xp_time):
    if xp_time == 'over_10':
        return df[df[category] > 10]
    elif xp_time == '5_to_10':
        return df[(df[category] > 5) & (df[category] <= 10)]
    elif xp_time == '1_to_5':
        return df[(df[category] > 1) & (df[category] <= 5)]
    else:
        return df[df[category] <= 1]

def create_correlation_plot(df_annotations, df_ground_truth, dorn_metric_value, buse_metric_value, llm_value, llm_name,
                            file_name=None):
    df_annotations = df_annotations.transpose()
    df_annotations.columns = df_annotations.iloc[0]  # Set the first row as column names
    df_annotations = df_annotations[1:]  # Remove the first row (now the header)
    df_annotations.columns = df_annotations.columns.astype(str).str.replace(r'\.0$', '', regex=True)
    df_annotations = df_annotations.rename_axis(None, axis=1).rename_axis('human_id', axis=0)
    correlations = []
    cores_nomeadas = ['red', 'orange', '#FFA500', '#9400D3']
    print('df_annotations=', df_annotations.shape[0], df_annotations.shape[1], '\n', df_annotations)
    print('df_ground_truth=', df_ground_truth.shape[0], df_ground_truth.shape[1], '\n', df_ground_truth)
    skipped = 0
    for annotator in df_annotations.columns:
        #Ensure that the indexes are the same in both dataframes
        merged_df = pd.merge(df_annotations[annotator], df_ground_truth, left_index=True, right_index=True, how='inner')
        # print('merged\n', merged_df)
        if len(merged_df.dropna()) >= 3:
            correlation, _ = spearmanr(merged_df[annotator], merged_df[0], nan_policy='omit')
            correlations.append(correlation)
        else:
            skipped = skipped + 1
    print(f"Skipping {skipped} due to insufficient data (less than 3 entries)")
    # Sort the correlations
    correlations.sort()

    # Create the plot
    plt.figure(figsize=(6, 6))  # Adjust figure size as needed
    plt.plot(correlations, color='black')

    # Add horizontal lines
    # plt.axhline(y=dorn_metric_value, color='#FFA500', linestyle='--', label=f'Dorn metric: {dorn_metric_value:.3f}')
    plt.axhline(y=np.median(correlations), color='blue', linestyle='-', label=f'median: {np.median(correlations):.3f}')
    plt.axhline(y=np.mean(correlations), color='blue', linestyle='--', label=f'avg: {np.mean(correlations):.3f}')
    # plt.axhline(y=buse_metric_value, color='#9400D3', linestyle='--', label=f'Buse metric: {buse_metric_value:.3f}')
    collor_count = 0
    sufix = ''
    if file_name in xp_times:
        sufix = file_name
    for name in llm_name:
        cor = cores_nomeadas[collor_count]
        collor_count = collor_count + 1
        plt.axhline(y=llm_value[name + sufix], color=cor, linestyle='-', label=f'{name} metric: {llm_value[name + sufix]:.3f}')

    # Add labels and title
    plt.xlabel('Annotators (sorted)')
    plt.ylabel('Spearman correlation with mean')
    plt.title('Spearman correlation with mean ' + sufix)
    plt.legend()
    plt.grid(False) #Remove grid to be more similar to the example
    plt.ylim(-1.1,1.1)
    plt.xlim(-100, len(correlations)+100)
    plt.tight_layout()
    plt.savefig(f'graficos/spearman_corr_{file_name}.png', dpi=300)
    # plt.show()

def calculate_icc(data):
    # Remove non-printable characters from the data
    # for key in data:
    #     data[key] = [str(item).replace('\u00A0','') for item in data[key]]

    # Convert numerical values back to float
    data["LLM Score Mean"] = [float(item) for item in data["LLM Score Mean"]]
    data["Human Score Mean"] = [float(item) for item in data["Human Score Mean"]]

    # Create DataFrame
    df = pd.DataFrame(data)
    # Prepare data for ICC calculation
    df_melted = df.reset_index().melt(id_vars=['index'], value_vars=['LLM Score Mean', 'Human Score Mean'], var_name='Rater', value_name='Score')
    # Calculate ICC
    icc = intraclass_corr(data=df_melted, targets='index', raters='Rater', ratings='Score')

    return icc

# A experiencia de cada humano conforme categoria Overall  Java  Cuda  Python  School  Industry
df_human_experience = loadHumanExperience()
print(df_human_experience)
# print(calcular_medianas(df_human_experience))
# print(calcular_medias(df_human_experience))

languages = ['cuda', 'java', 'python']
categories = ['Overall', 'Java', 'Cuda', 'Python', 'School', 'Industry']
xp_times = ['over_10', '5_to_10', '1_to_5', 'lt_1']
snippet_index = {'cuda':0, 'java':101, 'python':0}
df_human = {}
df_ground_truth = {}
for language in languages:
    # As notas que cada humano deu a cada arquivo (eles não deram notas a todos)
    df_human[language] = loadHumanScores(language, snippet_index[language])
    print(f'df_human[language-{language}]', df_human[language])
    # Os humanos, suas notas e seus atributos de experiência
    df_human[f'{language}_experience'] = merge_dataframes_by_human_id(df_human[language], df_human_experience)
    # print(f'df_human-experience{language}_experience ------------------->\n ', df_human[f'{language}_experience'])

    # Criamos um DF para cada grupo de humanos com uma mesma experiencia para cada linguagem
    for category in categories:
        for xp_time in xp_times:
            df_human[f'{language}_{category}_{xp_time}'] = filter(df_human[f'{language}_experience'], category, xp_time)
            if language.capitalize() == category and xp_time == 'over_10':
                print(f'df_human {language}_{category}_{xp_time}----------------------------- ', len(df_human[f'{language}_{category}_{xp_time}']))
                print(df_human[f'{language}_{category}_{xp_time}'])


# print(df_human)

df_llm = {}
llm_name = {}
df_analise = {}
df_analise['all_llm'] = pd.DataFrame()
llms_map = {'Gemini15flash': 'Gemini-1.5Flash', 'Gemini20flash':'Gemini-2.0Flash'}
for llm, llm_name_ in llms_map.items():
    df_llm[llm] = load('controlledDorn', llm)
    llm_name[llm] = llm_name_
    df_analise[llm] = pd.DataFrame(df_llm[llm].copy().transpose().mean().rename('LLM Score Mean'))

    for language in languages:
        for category in categories:
            for xp_time in xp_times:
                df_analise[f'{language}_{category}_{xp_time}'] = (df_analise[llm].copy().
                      join(df_human[f'{language}_{category}_{xp_time}'].mean().rename(f'{language}_{category}_{xp_time}'))).dropna()
                # print(f'{language}_{category}_{xp_time}----------------------------- ', len(df_analise[f'{language}_{category}_{xp_time}']))

    for language in languages:
        for category in categories:
            df_analise[f'{language}_{category}'] = (df_analise[llm].copy().
                  join(df_human[f'{language}_{category}_{xp_times[0]}'].mean().rename(f'{language}_{category}_{xp_times[0]}')).
                  join(df_human[f'{language}_{category}_{xp_times[1]}'].mean().rename(f'{language}_{category}_{xp_times[1]}')).
                  join(df_human[f'{language}_{category}_{xp_times[2]}'].mean().rename(f'{language}_{category}_{xp_times[2]}')).
                  join(df_human[f'{language}_{category}_{xp_times[3]}'].mean().rename(f'{language}_{category}_{xp_times[3]}'))).dropna()
    #         plot_ccorelacao_multiplos(
    #             [df_analise[f'{language}_{category}'][[f'{language}_{category}_{xp_times[3]}', f'{language}_{category}_{xp_times[0]}']],
    #                  df_analise[f'{language}_{category}'][[f'{language}_{category}_{xp_times[3]}', f'{language}_{category}_{xp_times[1]}']],
    #                  df_analise[f'{language}_{category}'][[f'{language}_{category}_{xp_times[2]}', f'{language}_{category}_{xp_times[0]}']],
    #                  df_analise[f'{language}_{category}'][[f'{language}_{category}_{xp_times[2]}', f'{language}_{category}_{xp_times[1]}']]],
    #             [(f'{language}_{category}_{xp_times[3]}', f'{language}_{category}_{xp_times[0]}'),
    #              (f'{language}_{category}_{xp_times[3]}', f'{language}_{category}_{xp_times[1]}'),
    #              (f'{language}_{category}_{xp_times[2]}', f'{language}_{category}_{xp_times[0]}'),
    #              (f'{language}_{category}_{xp_times[2]}', f'{language}_{category}_{xp_times[1]}')],
    #             f'humans_{language}_{category}')
    #         print(f'{language}_{category}-----------------------------', df_analise[f'{language}_{category}'])

    df_analise['all'] = df_analise[llm].copy()
    df_analise['all'] = df_analise['all'].copy().join(df_human[languages[0]].mean().rename('human_score'))
    df_analise['all'] = df_analise['all'].copy().merge(df_human[languages[1]].mean().rename('human_score'), left_index=True, right_index=True, how='outer')
    df_analise['all'] = df_analise['all'].copy().merge(df_human[languages[2]].mean().rename('human_score'), left_index=True, right_index=True, how='outer')
    df_analise['all']['Human Score Mean'] = df_analise['all']['human_score'].combine_first(df_analise['all']['human_score_y']).combine_first(df_analise['all']['human_score_x'])
    df_analise['all'] = df_analise['all'].drop(columns=['human_score', 'human_score_y', 'human_score_x'], errors='ignore')
    df_analise['all'] = df_analise['all'].drop('human_id')
    if df_analise['all_llm'].empty:
        df_analise['all_llm'] = df_analise['all'].rename(columns={'LLM Score Mean': llm}, inplace=False)
    else:
        df_analise['all_llm'] = df_analise['all_llm'].merge(df_analise['all'].rename(columns={'LLM Score Mean': llm}, inplace=False), how='outer')
    print(f'{llm}++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n',
      df_analise['all_llm'],
      # calculate_icc(df_analise['all']).to_string(),
      '\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    plot_correlacao(df_analise['all'], 'LLM Score Mean', 'Human Score Mean', llm)
    plot_violins(df_analise['all'], f'{llm}')

plot_violins(df_analise['all_llm'], 'all_llms')
# for language in languages:
#     for category in categories:
        # plot_violins(df_analise[f'{language}_{category}'], f'{language}_{category}')
        # plot_ccorelacao_multiplos([df_analise[f'{language}_{category}_{xp_times[0]}'], df_analise[f'{language}_{category}_{xp_times[1]}'],
        #        df_analise[f'{language}_{category}_{xp_times[2]}'], df_analise[f'{language}_{category}_{xp_times[3]}']],
        #       [('LLM Score Mean', f'{language}_{category}_{xp_times[0]}'), ('LLM Score Mean', f'{language}_{category}_{xp_times[1]}'),
        #        ('LLM Score Mean', f'{language}_{category}_{xp_times[2]}'), ('LLM Score Mean', f'{language}_{category}_{xp_times[3]}')],
        #               f'{language}_{category}')

# print(df_por_experiencia)
# plot_scatter(df_por_experiencia, 'LLM Score Mean', 'Graduate', '401')


# print(df_analise)
# print(analyze_scores(df_analise_todos, 'LLM Score Mean', 'Human Score Mean'))
# plot_ccorelacao(df_analise_todos, 'LLM Score Mean', 'Human Score Mean')
# plot_ccorelacao(df_analise_graduados, 'LLM Score Mean', 'Graduate')
# plot_scatter(df_analise, 'LLM Score Mean', 'Human Score Mean')
# plot_pairplot(df_analise, 'LLM Score Mean', 'Human Score Mean')
# plot_boxplots(df_analise)
# plot_violins(df_por_experiencia)

# Example DataFrames (replace with your actual data)

our_metric = 0.724
buse_metric = 0.309

# cohort = 'lt_1'
# cohort = '1_to_5'
# cohort = '5_to_10'
cohort = 'over_10'
for xp_time in xp_times:
    df_ground_truth[xp_time] = pd.DataFrame()
cohor = []
df_ground_truth['all'] = pd.DataFrame()
df_human['all'] = pd.DataFrame()
for language in languages:
    for xp_time in xp_times:
        if df_ground_truth[xp_time].empty:
            df_ground_truth[xp_time] = df_human[f'{language}_{language.capitalize()}_{xp_time}'].copy()
        else:
            df_ground_truth[xp_time] = df_ground_truth[xp_time].merge(df_human[f'{language}_{language.capitalize()}_{xp_time}'], on='human_id', how='outer')

    if df_ground_truth['all'].empty:
        df_ground_truth['all'] = df_human[language].copy()
    else:
        df_ground_truth['all'] = df_ground_truth['all'].merge(df_human[language], on='human_id', how='outer')

    if df_human['all'].empty:
        df_human['all'] = df_human[language].copy()
    else:
        df_human['all'] = df_human['all'].merge(df_human[language], on='human_id', how='outer')

df_ground_truth['mean'] = pd.DataFrame(df_ground_truth['all'].mean()).drop('human_id')
print('mean\n',df_ground_truth['mean'])
for xp_time in xp_times:
    print(f'Final {xp_time}\n', df_ground_truth[xp_time])
    df_ground_truth[f'{xp_time}_mean'] = pd.DataFrame(df_ground_truth[xp_time].mean()).drop(['human_id', 'Industry', 'School', 'Java', 'Cuda', 'Python', 'Overall'])
    print(f'{xp_time}\n', df_ground_truth[xp_time])
    print(f'{xp_time}_mean\n', df_ground_truth[f'{xp_time}_mean'])
print('all\n',df_human['all'])

llm_value = {}
llm_value_xp_time = {}
for llm_name_ in llm_name:
    print(df_analise[llm_name_]['LLM Score Mean'])
    llm_value[llm_name_] = df_analise[llm_name_]['LLM Score Mean'].corr(df_ground_truth['mean'][0], method='spearman')
    for xp_time in xp_times:
        llm_value_xp_time[llm_name_ + xp_time] = df_analise[llm_name_]['LLM Score Mean'].corr(df_ground_truth[f'{xp_time}_mean'][0], method='spearman')
#
# print('Correlação de Spearman, ', llm_value, ' com seniores ', llm_value_senior)

create_correlation_plot(df_human['all'].copy(), df_ground_truth['mean'].copy(), our_metric, buse_metric, llm_value, llm_name, 'llms_all')
#
# for xp_time in xp_times:
#     create_correlation_plot(df_human['all'].copy(), df_ground_truth[f'{xp_time}_mean'].copy(), our_metric, buse_metric, llm_value_xp_time, llm_name, xp_time)

# df_over_10 = pd.DataFrame(df_analise[llm_name]['LLM Score Mean']).join(pd.DataFrame(df_ground_truth[f'{cohort}_mean']),  how='outer').dropna()
# llm_value_senior = df_over_10['LLM Score Mean'].corr(df_over_10[0], method='spearman')
# print('llm_value_senior',llm_value_senior)
# print(df_over_10.to_string())
# create_correlation_plot(pd.DataFrame(df_over_10['LLM Score Mean']).copy(), pd.DataFrame(df_over_10[0]).copy(), our_metric, buse_metric, llm_value_senior)
