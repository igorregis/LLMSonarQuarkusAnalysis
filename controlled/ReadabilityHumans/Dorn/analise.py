import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import numpy as np

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

def plot_ccorelacao(df, column_one, column_two):
    sns.lmplot(x=column_one,y=column_two,data=df,aspect=2,height=6)
    plt.xlabel(column_one)
    plt.ylabel(column_two)
    plt.title(f'{column_one} Vs {column_two}')
    plt.savefig(f'graficos/correlacao-{column_one}-{column_two}.png', dpi=300)


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


df_human_experience = loadHumanExperience()
print(df_human_experience)
# print(calcular_medianas(df_human_experience))
# print(calcular_medias(df_human_experience))

languages = ['cuda', 'java', 'python']
categories = ['Overall', 'Java', 'Cuda', 'Python', 'School', 'Industry']
xp_times = ['over_10', '5_to_10', '1_to_5', 'lt_1']
snippet_index = {'cuda':0, 'java':101, 'python':0}
df_human = {}
for language in languages:
    df_human[language] = loadHumanScores(language, snippet_index[language])
    # print(df_human[language])
    df_human[f'{language}_experience'] = merge_dataframes_by_human_id(df_human[language], df_human_experience)
    print(df_human[f'{language}_experience'])

    for category in categories:
        for xp_time in xp_times:
            df_human[f'{language}_{category}_{xp_time}'] = filter(df_human[f'{language}_experience'], category, xp_time)
            print(f'{language}_{category}_{xp_time}----------------------------- ', len(df_human[f'{language}_{category}_{xp_time}']))
            print(df_human[f'{language}_{category}_{xp_time}'])

# print(df_human)

df_llm = load('controlledDorn', 'Gemini15flash')

print(df_llm)

df_analise = {}
df_analise['Gemini-1.5Flash'] = pd.DataFrame(df_llm.copy().transpose().mean().rename('LLM Score Mean'))
print(df_analise)

for language in languages:
    for category in categories:
        for xp_time in xp_times:
            df_analise[f'{language}_{category}_{xp_time}'] = (df_analise['Gemini-1.5Flash'].copy().
                  join(df_human[f'{language}_{category}_{xp_time}'].mean().rename(f'{language}_{category}_{xp_time}'))).dropna()
            print(f'{language}_{category}_{xp_time}----------------------------- ', len(df_analise[f'{language}_{category}_{xp_time}']))

# for language in languages:
#     for category in categories:
#         df_analise[f'{language}_{category}'] = (df_analise['Gemini-1.5Flash'].copy().
#               join(df_human[f'{language}_{category}_{xp_times[0]}'].mean().rename(f'{language}_{category}_{xp_times[0]}')).
#               join(df_human[f'{language}_{category}_{xp_times[1]}'].mean().rename(f'{language}_{category}_{xp_times[1]}')).
#               join(df_human[f'{language}_{category}_{xp_times[2]}'].mean().rename(f'{language}_{category}_{xp_times[2]}')).
#               join(df_human[f'{language}_{category}_{xp_times[3]}'].mean().rename(f'{language}_{category}_{xp_times[3]}'))).dropna()
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


for language in languages:
    for category in categories:
        plot_violins(df_analise[f'{language}_{category}'], f'{language}_{category}')
        plot_ccorelacao_multiplos([df_analise[f'{language}_{category}_{xp_times[0]}'], df_analise[f'{language}_{category}_{xp_times[1]}'],
               df_analise[f'{language}_{category}_{xp_times[2]}'], df_analise[f'{language}_{category}_{xp_times[3]}']],
              [('LLM Score Mean', f'{language}_{category}_{xp_times[0]}'), ('LLM Score Mean', f'{language}_{category}_{xp_times[1]}'),
               ('LLM Score Mean', f'{language}_{category}_{xp_times[2]}'), ('LLM Score Mean', f'{language}_{category}_{xp_times[3]}')],
                      f'{language}_{category}')

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



