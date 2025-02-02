import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr

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

def plot_violins(df, llm):
    df_melted = df.melt(var_name='Column', value_name='Value')

    plt.figure(figsize=(12, 9))
    sns.violinplot(x='Column', y='Value', data=df_melted, orient='v')
    plt.title('Violin Plots for All Columns')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.savefig(f'graficos/violinos_{llm}.png', dpi=300)

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


def plot_ccorelacao_multiplos(dfs, columns, llm):
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
    plt.savefig(f'graficos/correlacao_{llm}.png', dpi=300)

def plot_scatter(df, column_one, column_two, column_three):
    sns.scatterplot(x=column_one, y=column_two, hue=column_three, data=df)
    sns.regplot(x=column_one, y=column_two, data=df, scatter=False, color='blue')
    plt.xlabel(column_one)
    plt.ylabel(column_two)
    plt.title(f'{column_one} Vs {column_two} com {column_three}')
    plt.savefig(f'graficos/scatter-{column_one}-{column_two}-{column_three}.png', dpi=300)

def plot_correlacao(df, column_one, column_two, llm):
    fig, ax = plt.subplots(figsize=(12, 8))  # Ajuste o tamanho aqui (largura, altura)

    sns.regplot(x=column_one, y=column_two, data=df, ax=ax, scatter_kws={'alpha':0.7}) # Usar regplot e ax
    # Calculando a inclinação da linha de tendência
    x = df[column_one]
    y = df[column_two]
    slope, intercept = np.polyfit(x, y, 1)
    angle = np.degrees(np.arctan(slope))

    ax.set_xlabel(column_one)  # Configurar os labels nos eixos
    ax.set_ylabel(column_two)
    ax.set_title(f'{column_one}({llm}) Vs {column_two} - angle: {angle:.2f}°')

    plt.tight_layout()  # Ajusta o layout para evitar cortes
    plt.savefig(f'graficos/correlacao-{llm}-{column_one}-{column_two}.png', dpi=300)
    plt.close(fig)  # Fecha a figura para liberar memória

def create_correlation_plot(df_annotations, df_ground_truth, dorn_metric_value, buse_metric_value, llm_value,
                            file_name=None):
    df_annotations = df_annotations.transpose()
    df_annotations.columns = df_annotations.iloc[0]  # Set the first row as column names
    df_annotations = df_annotations[1:]  # Remove the first row (now the header)
    df_annotations = df_annotations.drop('course_code')
    df_annotations.columns = df_annotations.columns.astype(str).str.replace(r'\.0$', '', regex=True)
    df_annotations = df_annotations.rename_axis(None, axis=1).rename_axis('human_id', axis=0)
    correlations = []
    cores_nomeadas = ['red', 'orange', '#FFA500', '#9400D3']
    print('df_annotations=',df_annotations.shape[0], df_annotations.shape[1], '\n', df_annotations)
    print('df_ground_truth=',df_ground_truth.shape[0], df_ground_truth.shape[1], '\n', df_ground_truth)
    skipped = 0
    for annotator in df_annotations.columns:
        #Ensure that the indexes are the same in both dataframes
        merged_df = pd.merge(df_annotations[annotator], df_ground_truth, left_index=True, right_index=True, how='inner')
        # print('merged\n', merged_df)
        if len(merged_df.dropna()) >= 3:
            merged_df[annotator] = pd.to_numeric(merged_df[annotator], errors='coerce')
            correlation, _ = spearmanr(merged_df[annotator], merged_df['Human Score Mean'], nan_policy='omit')
            # correlation, _ = pearsonr(merged_df[annotator], merged_df['Human Score Mean'])
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
    for name in llms:
        cor = cores_nomeadas[collor_count]
        collor_count = collor_count + 1
        print(name + sufix)
        plt.axhline(y=llm_value[name + sufix], color=cor, linestyle='-', label=f'{name} metric: {llm_value[name + sufix]:.3f}')

    # Add labels and title
    plt.xlabel('Annotators (sorted)')
    plt.ylabel('Spearman correlation with mean')
    plt.title('Spearman correlation with mean ' + sufix)
    plt.legend()
    plt.grid(False) #Remove grid to be more similar to the example
    plt.ylim(-1.1,1.1)
    plt.xlim(0, len(correlations)+1)
    plt.tight_layout()
    plt.savefig(f'graficos/spearman_corr_{file_name}.png', dpi=300)
    # plt.show()

xp_times = ['101', '201', '401', 'graduate']

df_human = loadHumanScores("oracle.csv")

df_cs101 = df_human[df_human['course_code'].isin(['cs101', 'cs101e'])]
df_cs201 = df_human[df_human['course_code'].isin(['cs201-Software_Dev', 'cs216-Prog_Data_Rep'])]
df_cs401 = df_human[df_human['course_code'].isin(['cs414-OS', 'cs445-Intro_Graphics'])]
df_graduate = df_human[df_human['course_code'].isin(['graduate'])]

df_all = df_human.drop(columns=['student_id', 'course_code'])
df_cs101 = df_cs101.drop(columns=['student_id', 'course_code'])
df_cs201 = df_cs201.drop(columns=['student_id', 'course_code'])
df_cs401 = df_cs401.drop(columns=['student_id', 'course_code'])
df_graduate = df_graduate.drop(columns=['student_id', 'course_code'])

print(df_human)

df_llm = {}
llms = ['Gemini15flash', 'Gemini20flash']
for llm in llms:
    df_llm[llm] = load('controlledBuseAndWeimer', llm)
print(df_llm)

df_analise = {}
df_analise_todos = {}
df_analise_101 = {}
df_analise_201 = {}
df_analise_401 = {}
df_analise_graduados = {}
df_por_experiencia = {}
llm_value = {}
for llm in llms:
    df_analise[llm] = pd.DataFrame(df_llm[llm].copy().transpose().mean().rename('LLM Score Mean'))
    df_analise_todos[llm] = df_analise[llm].copy().join(df_all.copy().mean().rename('Human Score Mean'))
    df_analise_101[llm] = df_analise[llm].copy().join(df_cs101.copy().mean().rename('101'))
    df_analise_201[llm] = df_analise[llm].copy().join(df_cs201.copy().mean().rename('201'))
    df_analise_401[llm] = df_analise[llm].copy().join(df_cs401.copy().mean().rename('401'))
    df_analise_graduados[llm] = df_analise[llm].copy().join(df_graduate.copy().mean().rename('Graduate'))

    df_por_experiencia[llm] = (df_analise[llm].join(df_graduate.copy().mean().rename('Graduate')).
       join(df_cs401.copy().mean().rename('401')).join(df_cs201.copy().mean().rename('201')).join(df_cs101.copy().mean().rename('101')))

    # plot_violins(df_por_experiencia)

    print(df_por_experiencia[llm])
    # plot_scatter(df_por_experiencia, 'LLM Score Mean', 'Graduate', '401')

    plot_ccorelacao_multiplos([df_analise_graduados[llm], df_analise_401[llm], df_analise_201[llm], df_analise_101[llm]],
                              [('LLM Score Mean', 'Graduate'), ('LLM Score Mean', '401'),
                               ('LLM Score Mean', '201'), ('LLM Score Mean', '101')], llm)

    # print(df_analise)
    print(llm, analyze_scores(df_analise_todos[llm], 'LLM Score Mean', 'Human Score Mean'))
    # plot_ccorelacao(df_analise_todos, 'LLM Score Mean', 'Human Score Mean')
    # plot_ccorelacao(df_analise_graduados, 'LLM Score Mean', 'Graduate')
    # plot_scatter(df_analise, 'LLM Score Mean', 'Human Score Mean')
    # plot_pairplot(df_analise, 'LLM Score Mean', 'Human Score Mean')
    # plot_boxplots(df_analise)
    plot_violins(df_por_experiencia[llm], llm)
    plot_correlacao(df_analise_todos[llm], 'LLM Score Mean', 'Human Score Mean', llm)
    llm_value[llm] = df_analise_todos[llm]['LLM Score Mean'].corr(df_analise_todos[llm]['Human Score Mean'], method='spearman')
    print(llm_value[llm])
    # print(df_analise_todos[llm])

df_dois_llm = df_analise_todos[llms[0]].copy().drop(columns=['LLM Score Mean'])
df_ground_truth = df_dois_llm.copy()
df_dois_llm = df_dois_llm.drop(columns=['Human Score Mean'])
print('df_ground_truth', df_ground_truth)
df_dois_llm[llms[0]] = df_analise_todos[llms[0]]['LLM Score Mean']
df_dois_llm[llms[1]] = df_analise_todos[llms[1]]['LLM Score Mean']
plot_correlacao(df_dois_llm, llms[0], llms[1], 'llms')
print(df_dois_llm[llms[1]].corr(df_dois_llm[llms[0]], method='spearman'))
print(df_dois_llm)

for llm in llms:
    create_correlation_plot(df_human, df_ground_truth, 0,0, llm_value, 'llms')