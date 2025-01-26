import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import numpy as np

def loadHumanScores(file_path):
    df = pd.read_csv(file_path, header=None)
    snippet_columns = [f'snippets/{i}.jsnp' for i in range(1, len(df.columns) - 1)]
    df.columns = ['student_id', 'course_code'] + snippet_columns
    return df

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

def plot_violins(df):
    df_melted = df.melt(var_name='Column', value_name='Value')

    plt.figure(figsize=(12, 9))
    sns.violinplot(x='Column', y='Value', data=df_melted, orient='v')
    plt.title('Violin Plots for All Columns')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.savefig(f'graficos/violinos_TODOS.png', dpi=300)

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


def plot_ccorelacao_multiplos(dfs, columns):
    num_plots = len(dfs)
    num_cols = 2
    num_rows = (num_plots + 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 6 * num_rows))
    axes = axes.flatten()

    for ax, (df, (column_one, column_two)) in zip(axes, zip(dfs, columns)):
        sns.scatterplot(x=column_one, y=column_two, data=df, ax=ax)
        sns.regplot(x=column_one, y=column_two, data=df, scatter=False, ax=ax, color='blue')

        # Calculando a inclinação da linha de tendência
        x = df[column_one]
        y = df[column_two]
        slope, intercept = np.polyfit(x, y, 1)
        angle = np.degrees(np.arctan(slope))

        ax.set_xlabel(column_one)
        ax.set_ylabel(column_two)
        ax.set_title(f'{column_one} Vs {column_two} Ângulo de inclinação: {angle:.2f}°')

    # Remover subgráficos vazios
    for ax in axes[num_plots:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(f'graficos/correlacao_TODOS.png', dpi=300)

def plot_scatter(df, column_one, column_two, column_three):
    sns.scatterplot(x=column_one, y=column_two, hue=column_three, data=df)
    sns.regplot(x=column_one, y=column_two, data=df, scatter=False, color='blue')
    plt.xlabel(column_one)
    plt.ylabel(column_two)
    plt.title(f'{column_one} Vs {column_two} com {column_three}')
    plt.savefig(f'graficos/scatter-{column_one}-{column_two}-{column_three}.png', dpi=300)

df_human = loadHumanScores("oracle.csv")
df_llm = load('controlledBuseAndWeimer', 'Gemini15flash')

df_cs101 = df_human[df_human['course_code'].isin(['cs101', 'cs101e'])]
df_cs201 = df_human[df_human['course_code'].isin(['cs201-Software_Dev', 'cs216-Prog_Data_Rep'])]
df_cs401 = df_human[df_human['course_code'].isin(['cs414-OS', 'cs445-Intro_Graphics'])]
df_graduate = df_human[df_human['course_code'].isin(['graduate'])]

df_cs101 = df_cs101.drop(columns=['student_id', 'course_code'])
df_cs201 = df_cs201.drop(columns=['student_id', 'course_code'])
df_cs401 = df_cs401.drop(columns=['student_id', 'course_code'])
df_graduate = df_graduate.drop(columns=['student_id', 'course_code'])

print(df_human)
print(df_llm)

df_temp = df_human.drop(columns=['student_id', 'course_code'])


df_analise = pd.DataFrame(df_llm.copy().transpose().mean().rename('LLM Score Mean'))
df_analise_todos = df_analise.copy().join(df_temp.mean().rename('Human Score Mean'))
df_analise_101 = df_analise.copy().join(df_cs101.mean().rename('101'))
df_analise_201 = df_analise.copy().join(df_cs201.mean().rename('201'))
df_analise_401 = df_analise.copy().join(df_cs401.mean().rename('401'))
df_analise_graduados = df_analise.copy().join(df_graduate.mean().rename('Graduate'))

df_por_experiencia = (df_analise.join(df_graduate.mean().rename('Graduate')).join(df_cs401.mean().rename('401')).
                      join(df_cs201.mean().rename('201')).join(df_cs101.mean().rename('101')))

# plot_violins(df_por_experiencia)

print(df_por_experiencia)
# plot_scatter(df_por_experiencia, 'LLM Score Mean', 'Graduate', '401')

plot_ccorelacao_multiplos([df_analise_graduados, df_analise_401, df_analise_201, df_analise_101],
                          [('LLM Score Mean', 'Graduate'), ('LLM Score Mean', '401'),
                           ('LLM Score Mean', '201'), ('LLM Score Mean', '101')])

# print(df_analise)
print(analyze_scores(df_analise_todos, 'LLM Score Mean', 'Human Score Mean'))
# plot_ccorelacao(df_analise_todos, 'LLM Score Mean', 'Human Score Mean')
# plot_ccorelacao(df_analise_graduados, 'LLM Score Mean', 'Graduate')
# plot_scatter(df_analise, 'LLM Score Mean', 'Human Score Mean')
# plot_pairplot(df_analise, 'LLM Score Mean', 'Human Score Mean')
# plot_boxplots(df_analise)
plot_violins(df_por_experiencia)



