import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import pearsonr

def loadHumanScores(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=1)
    snippet_columns = [f'snippets/{i}.jsnp' for i in range(1, len(df.columns))]
    df.columns = ['student_id'] + snippet_columns
    return df

def loadScalabrino(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=1)
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

def create_correlation_plot(df_annotations, df_ground_truth, scalabrino_value, llm_value,
                            file_name=None):
    df_annotations = df_annotations.transpose()
    # print(df_annotations)
    df_annotations.columns = df_annotations.iloc[0]  # Set the first row as column names
    df_annotations = df_annotations[1:]  # Remove the first row (now the header)
    # df_annotations = df_annotations.drop('student_id')
    df_annotations.columns = df_annotations.columns.astype(str).str.replace(r'\.0$', '', regex=True)
    df_annotations = df_annotations.rename_axis(None, axis=1).rename_axis('human_id', axis=0)
    correlations = []
    cores_nomeadas = ['red', 'orange', '#FFA500', '#9400D3']
    # print('df_annotations=',df_annotations.shape[0], '\n', df_annotations)
    # print('df_ground_truth=',df_ground_truth.shape[0], '\n', df_ground_truth)
    skipped = 0
    for annotator in df_annotations.columns:
        #Ensure that the indexes are the same in both dataframes
        merged_df = pd.merge(df_annotations[annotator], df_ground_truth, left_index=True, right_index=True, how='inner')
        # print('merged\n', merged_df)
        if len(merged_df.dropna()) >= 3:
            merged_df[annotator] = pd.to_numeric(merged_df[annotator], errors='coerce')
            correlation, _ = spearmanr(merged_df[annotator], merged_df['Humans'], nan_policy='omit')
            # correlation, _ = pearsonr(merged_df[annotator], merged_df['Humans'])
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
    plt.axhline(y=scalabrino_value, color='#9400D3', linestyle='--', label=f'Scalabrino metric: {scalabrino_value:.3f}')
    plt.axhline(y=np.median(correlations), color='blue', linestyle='-', label=f'median: {np.median(correlations):.3f}')
    plt.axhline(y=np.mean(correlations), color='blue', linestyle='--', label=f'avg: {np.mean(correlations):.3f}')
    # plt.axhline(y=buse_metric_value, color='#9400D3', linestyle='--', label=f'Buse metric: {buse_metric_value:.3f}')
    collor_count = 0
    for name in llms:
        cor = cores_nomeadas[collor_count]
        collor_count = collor_count + 1
        plt.axhline(y=llm_value[name], color=cor, linestyle='-', label=f'{name} metric: {llm_value[name]:.3f}')

    # Add labels and title
    plt.xlabel('Annotators (sorted)')
    plt.ylabel('Spearman correlation with mean')
    plt.title('Spearman correlation with mean')
    plt.legend()
    plt.grid(False) #Remove grid to be more similar to the example
    plt.ylim(-1.1,1.1)
    plt.xlim(0, len(correlations)-1)
    plt.tight_layout()
    plt.savefig(f'graficos/spearman_corr_{file_name}.png', dpi=300)
    # plt.show()

def perform_mcnemar_test(df1, df2, alpha=0.05):
    try:
        # Create contingency table from the DataFrames.  We need to align how the
        # confusion matrix is structured for McNemar's test.  The rows and columns
        # represent the predictions of the two models.
        n00 = sum((df1['TP'] + df1['TN']) & (df2['TP'] + df2['TN'])) # Both models agree it is readable or non-readable
        n01 = sum((df1['TP'] + df1['TN']) & (df2['FP'] + df2['FN'])) # Model 1 is correct, model 2 is wrong
        n10 = sum((df1['FP'] + df1['FN']) & (df2['TP'] + df2['TN'])) # Model 1 is wrong, model 2 is correct
        n11 = sum((df1['FP'] + df1['FN']) & (df2['FP'] + df2['FN'])) # Both models are wrong


        contingency_table = [[n00, n01], [n10, n11]]

        # Perform McNemar's test
        result = mcnemar(contingency_table, correction=True)  # Apply continuity correction

        return {"statistic": result.statistic, "pvalue": result.pvalue}

    except (KeyError, TypeError, ValueError) as e:
        print(f"Error processing DataFrames: {e}")
        return None

def compute_sccuracy(df_scalabrino_analise):
    scalabrino = pd.DataFrame()
    llm1 = pd.DataFrame()
    llm2 = pd.DataFrame()
    llm3 = pd.DataFrame()
    scalabrino['TP'] = ((df_scalabrino_analise['Readable for Scalabrino'] == 1) & (df_scalabrino_analise['Readable for Human'] == 1)).astype(int)
    scalabrino['TN'] = ((df_scalabrino_analise['Readable for Scalabrino'] == 0) & (df_scalabrino_analise['Readable for Human'] == 0)).astype(int)
    scalabrino['FP'] = ((df_scalabrino_analise['Readable for Scalabrino'] == 1) & (df_scalabrino_analise['Readable for Human'] == 0)).astype(int)
    scalabrino['FN'] = ((df_scalabrino_analise['Readable for Scalabrino'] == 0) & (df_scalabrino_analise['Readable for Human'] == 1)).astype(int)
    print('+++++++++++++++++++++++++++++++++++++++++++++++\n\nScalabrino accuracy: ',
          (scalabrino['TP'].sum() + scalabrino['TN'].sum())/(scalabrino['TP'].sum()+scalabrino['TN'].sum()+scalabrino['FP'].sum()+scalabrino['FN'].sum()))

    llm1['TP'] = ((df_scalabrino_analise['Readable for Gemini15flash'] == 1) & (df_scalabrino_analise['Readable for Human'] == 1)).astype(int)
    llm1['TN'] = ((df_scalabrino_analise['Readable for Gemini15flash'] == 0) & (df_scalabrino_analise['Readable for Human'] == 0)).astype(int)
    llm1['FP'] = ((df_scalabrino_analise['Readable for Gemini15flash'] == 1) & (df_scalabrino_analise['Readable for Human'] == 0)).astype(int)
    llm1['FN'] = ((df_scalabrino_analise['Readable for Gemini15flash'] == 0) & (df_scalabrino_analise['Readable for Human'] == 1)).astype(int)
    print('+++++++++++++++++++++++++++++++++++++++++++++++\n\nGemini15flash accuracy: ',
          (llm1['TP'].sum() + llm1['TN'].sum())/(llm1['TP'].sum()+llm1['TN'].sum()+llm1['FP'].sum()+llm1['FN'].sum()))

    llm2['TP'] = ((df_scalabrino_analise['Readable for Gemini20flash'] == 1) & (df_scalabrino_analise['Readable for Human'] == 1)).astype(int)
    llm2['TN'] = ((df_scalabrino_analise['Readable for Gemini20flash'] == 0) & (df_scalabrino_analise['Readable for Human'] == 0)).astype(int)
    llm2['FP'] = ((df_scalabrino_analise['Readable for Gemini20flash'] == 1) & (df_scalabrino_analise['Readable for Human'] == 0)).astype(int)
    llm2['FN'] = ((df_scalabrino_analise['Readable for Gemini20flash'] == 0) & (df_scalabrino_analise['Readable for Human'] == 1)).astype(int)
    print('+++++++++++++++++++++++++++++++++++++++++++++++\n\nGemini20flash accuracy: ',
          (llm2['TP'].sum() + llm2['TN'].sum())/(llm2['TP'].sum()+llm2['TN'].sum()+llm2['FP'].sum()+llm2['FN'].sum()))

    llm3['TP'] = ((df_scalabrino_analise['Readable for Gemini25flash-lite-thinking'] == 1) & (df_scalabrino_analise['Readable for Human'] == 1)).astype(int)
    llm3['TN'] = ((df_scalabrino_analise['Readable for Gemini25flash-lite-thinking'] == 0) & (df_scalabrino_analise['Readable for Human'] == 0)).astype(int)
    llm3['FP'] = ((df_scalabrino_analise['Readable for Gemini25flash-lite-thinking'] == 1) & (df_scalabrino_analise['Readable for Human'] == 0)).astype(int)
    llm3['FN'] = ((df_scalabrino_analise['Readable for Gemini25flash-lite-thinking'] == 0) & (df_scalabrino_analise['Readable for Human'] == 1)).astype(int)
    print('+++++++++++++++++++++++++++++++++++++++++++++++\n\nGemini25flash-lite-thinking accuracy: ',
          (llm3['TP'].sum() + llm3['TN'].sum()) / (llm3['TP'].sum() + llm3['TN'].sum() + llm3['FP'].sum() + llm3['FN'].sum()))

    mcnemar_results = perform_mcnemar_test(scalabrino, llm1)
    if mcnemar_results:
        print(f"McNemar's test statistic: {mcnemar_results['statistic']}")
        print(f"P-value: {mcnemar_results['pvalue']}")

        alpha = 0.05  # Significance level
        if mcnemar_results['pvalue'] < alpha:
            print("The difference between the models is statistically significant.")
        else:
            print("The difference between the models is not statistically significant.")

    dfs = [scalabrino, llm1, llm2, llm3]
    comparisons = [(0, 1), (0, 2), (1, 2)]  # Indices for the comparisons
    p_values = []
    for i, j in comparisons:
        results = perform_mcnemar_test(dfs[i], dfs[j])
        if results:
            p_values.append(results['pvalue'])
        else:
            print(f"Error in comparison between DataFrame {i + 1} and {j + 1}")  # Error message remains

    # Bonferroni correction
    from statsmodels.sandbox.stats.multicomp import multipletests
    corrected_p_values = multipletests(p_values, method='bonferroni')[1]

    for k, (i, j) in enumerate(comparisons):  # Use enumerate to get an index k
        print(f"Comparison DataFrame {i + 1} vs {j + 1}:")  # This line is correct now
        print(f"Original p-value: {p_values[k]:.3f}")  # Access with correct index
        print(f"Corrected p-value: {corrected_p_values[k]:.3f}")  # Access with correct index

        if corrected_p_values[k] < alpha:  # Access with correct index
            print("Statistically significant after Bonferroni correction.")
        else:
            print("Not statistically significant after Bonferroni correction.")


df_human = loadHumanScores("scores.csv")

df_all = df_human.drop(columns=['student_id'])

print(df_human)

df_llm = {}
llms = ['Gemini15flash', 'Gemini20flash', 'Gemini25flash-lite-thinking']
for llm in llms:
    df_llm[llm] = load('controlledScalabrino', llm)
    df_llm[llm + 'transposed'] = df_llm[llm].transpose()
    print(llm + 'transposed\n', df_llm[llm + 'transposed'])
    df_llm[llm + 'mean'] = pd.DataFrame(df_llm[llm + 'transposed'].mean().rename(llm))
    print(llm + 'mean\n', df_llm[llm + 'mean'])


df_ground_truth = pd.DataFrame(df_human.drop(columns='student_id').mean().rename('Humans'))
df_ground_truth['Readable for Human'] = (df_ground_truth['Humans'] > 3.6).astype(int)
print(df_ground_truth)

df_analise = df_ground_truth.copy()
for llm in llms:
    df_analise = df_analise.join(df_llm[llm + 'mean'])
    df_analise[f'Readable for {llm}'] = (df_llm[llm + 'mean'] > 3.6).astype(int)
    # print(df_analise.to_string())

print(df_analise)

df_scalabrino = loadScalabrino('scalabrino.csv')
df_scalabrino.columns = ['name', 'Scalabrino Score']
df_scalabrino['Scalabrino Scaled Score'] = df_scalabrino['Scalabrino Score']*100/20
df_scalabrino['Readable for Scalabrino'] = (df_scalabrino['Scalabrino Scaled Score'] > 3.6).astype(int)
df_scalabrino = df_scalabrino.set_index('name')
print('Scalabrino\n', df_scalabrino)
df_scalabrino_analise = df_scalabrino.join(df_ground_truth)
print('Scalabrino Analise\n', df_scalabrino_analise)

df_analise_all = df_scalabrino_analise.drop(columns=['Humans', 'Readable for Human']).join(df_analise)
print('df_analise_all\n',df_analise_all.to_string())
print(df_analise_all['Humans'].mean())

compute_sccuracy(df_analise_all)

scalabrino_value = df_scalabrino_analise['Scalabrino Score'].corr(df_scalabrino_analise['Humans'])
print("Correlação ", scalabrino_value)

llm_value = {}
for llm_name in llms:
    # print(df_analise[llm_name])
    llm_value[llm_name] = df_analise[llm_name].corr(df_ground_truth['Humans'], method='spearman')

create_correlation_plot(df_human.copy(), df_ground_truth['Humans'].copy(),scalabrino_value, llm_value, 'llms_all')

