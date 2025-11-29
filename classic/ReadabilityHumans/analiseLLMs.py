from typing import Optional
import pingouin as pg
import krippendorff
import pandas as pd
from factor_analyzer import FactorAnalyzer
import numpy as np

import warnings
# Isso irá filtrar o FutureWaning do scikit-learn
warnings.filterwarnings("ignore", category=FutureWarning)


def loadHumanScores(language, sinippet_init):
    df = pd.read_csv(f'Dorn/{language.lower()}.csv', header=None)
    snippet_columns = [f'snippets/{language}/{i}.jsnp' for i in range(sinippet_init, len(df.columns) + sinippet_init - 1)]
    df.columns = ['name'] + snippet_columns
    df.columns = df.columns.str.replace(f'snippets/{language}/', '')
    return df.transpose()

def loadScalabrinoHumanScores(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=1)
    snippet_columns = [f'snippets/{i}.jsnp' for i in range(1, len(df.columns))]
    df.columns = ['student_id'] + snippet_columns
    df.columns = df.columns.str.replace('snippets/', '')
    snippet_cols_to_multiply = [
        col for col in df.columns if col.endswith('.jsnp')
    ]
    df[snippet_cols_to_multiply] = df[snippet_cols_to_multiply].astype(float) * 20 # Para Transformar em escala 0 a 100
    df = df.rename(columns={'student_id': 'name'})
    df = df.set_index('name').transpose()
    df = df.sort_index()
    return df

def loadScalabrino(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=1)
    df.columns = ['name', 'Scalabrino Score']
    df['name'] = df['name'].str.replace('snippets/', '')
    df['Scalabrino Score'] = df['Scalabrino Score'].astype(float) * 100
    df = df.set_index('name')
    df = df.sort_index()
    return df


def loadLLMScores(llm, dataset):
    dfs = []

    for n in range(1, 4):
        file_path = f"{llm}/classic{dataset}{llm}-{n}.json"

        # lines=True é crucial pois o formato apresentado é NDJSON (um objeto por linha)
        try:
            df_temp = pd.read_json(file_path, lines=True)
        except ValueError:
            df_temp = pd.read_json(file_path)

        df_temp = df_temp[['name', 'score']]
        df_temp['score'] = df_temp['score'].astype(float) #/ 2 #Para carregar em escala likert 1 a 10 dividimos por 2
        col_name = f"{llm}-{n}"
        df_temp = df_temp.rename(columns={'score': col_name})

        # Define 'name' como índice para garantir o alinhamento correto na concatenação
        df_temp = df_temp.set_index('name')

        dfs.append(df_temp)

    # Concatena todos os DataFrames horizontalmente (axis=1) alinhando pelo índice (name)
    df = pd.concat(dfs, axis=1)
    df = df.sort_index()
    return df


def calculate_mean_scores(df_llm_scores):
    """
    Calcula a média de todas as colunas de score do LLM para cada snippet (linha),
    e retorna um novo DataFrame com o nome do snippet (índice) e a média calculada.

    Args:
        df_llm_scores (pd.DataFrame): DataFrame com o nome do snippet como índice
                                     e colunas de score (e.g., 'Gemini25flash-lite-thinking-1').

    Returns:
        pd.DataFrame: Novo DataFrame com o índice original e a coluna de média ('LLM_Mean_Score').
    """

    # 1. Verificar se o DataFrame tem o índice nomeado 'name'
    if df_llm_scores.index.name != 'name':
        # Se o índice não tiver nome, tentamos inferir. No seu caso, o DF LLM já vem com o nome 'name'.
        # Se for o caso do DF 2 (Human), você pode precisar de uma lógica de renomeação aqui,
        # mas estamos focando no DF LLM.
        pass  # Mantém o índice como está

    # 2. Calcular a média de todas as colunas numéricas (eixo das colunas, axis=1).
    # Como todas as colunas no seu DF (Gemini-1, Gemini-2, ...) são de score,
    # o .mean(axis=1) funcionará corretamente em todas elas.
    mean_series = df_llm_scores.mean(axis=1)

    # 3. Criar o novo DataFrame
    # O Series resultante do .mean(axis=1) já tem o índice (os nomes dos snippets).
    # O nome da nova coluna será o mesmo do LLM principal (sem o -N) mais o sufixo "Mean_Score"

    # Inferir o nome base do LLM (e.g., 'Gemini25flash-lite-thinking') a partir da primeira coluna
    first_col_name = df_llm_scores.columns[0]
    llm_base_name = first_col_name.rsplit('-', 1)[0]
    new_col_name = f"{llm_base_name}"

    # Cria o DataFrame de resultado a partir da Series da média
    df_result = mean_series.to_frame(name=new_col_name)

    # 4. Opcional: Renomear o índice para garantir uniformidade na saída,
    # se o nome do índice foi perdido. Mas, no seu caso, ele já está como 'name'.
    df_result.index.name = 'name'

    return df_result


def calculate_mcdonalds_omega_overall(df: pd.DataFrame) -> float or None:
    """
    Calcula o Omega de McDonald (Total) para avaliar consistência interna.

    Suposição de Entrada:
      - Linhas (Index): Observações / Runs / Sujeitos
      - Colunas: Itens / Variáveis / Perguntas
    """
    # Garante que é numérico
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    # Remove linhas que sejam totalmente NaN (runs falhadas)
    df_clean = df_numeric.dropna(axis=0, how='all')

    # Remove colunas que sejam totalmente NaN (itens inválidos)
    df_clean = df_clean.dropna(axis=1, how='all')

    n_runs, n_items = df_clean.shape

    # 1. Checagem de Variância Zero (Estabilidade Perfeita vs Falha de Instrumento)
    # Se o desvio padrão de TODOS os itens for 0, não há variância.
    # No contexto de LLMs: Significa que a LLM gerou sempre a mesma saída.
    std_devs = df_clean.std(axis=0)
    if (std_devs < 1e-9).all():
        print("ℹ️ Variância Zero em todos os itens: Estabilidade perfeita detectada.")
        return 1.0

    # 2. Checagem de Tamanho Mínimo para Análise Fatorial
    # FA requer matriz de correlação invertível. N < 3 ou Itens < 3 geralmente quebram.
    if n_runs < 5 or n_items < 3:
        print(f"⚠️ Dados insuficientes para Análise Fatorial (Runs={n_runs}, Itens={n_items}).")
        # Retornar None é mais seguro que 1.0 aqui, pois indica impossibilidade de cálculo
        # e não 'perfeição'.
        return None

        # 3. Tratamento de Colunas sem Variância (Constantes)
    # Colunas com variância zero geram erros na Análise Fatorial.
    # Precisamos removê-las, mas isso altera o construto analisado.
    cols_with_variance = std_devs[std_devs > 1e-9].index
    df_final = df_clean[cols_with_variance]

    if df_final.shape[1] < 3:
        print("⚠️ Após remover itens constantes, restaram menos de 3 itens.")
        return None

    try:
        # 4. Análise Fatorial (1 Fator)
        # check_input=False evita que o pacote lance erro antes de tentarmos tratar
        fa = FactorAnalyzer(n_factors=1, rotation=None, method='minres')
        # print("df_final\n",df_final)
        fa.fit(df_final)

        loadings = fa.loadings_
        error_variances = fa.get_uniquenesses()

        # 5. Cálculo do Omega
        sum_loadings = np.sum(loadings)
        sum_sq_loadings = sum_loadings ** 2
        sum_error = np.sum(error_variances)

        denominator = sum_sq_loadings + sum_error

        if denominator == 0:
            return 0.0  # Evita divisão por zero

        omega = sum_sq_loadings / denominator
        return float(omega)

    except Exception as e:
        error_msg = str(e).lower()
        # Se a matriz for singular, significa que as colunas são linearmente dependentes
        # (Runs muito parecidas ou idênticas) -> Consistência Perfeita.
        if "singular matrix" in error_msg:
            # print("ℹ️ Matriz Singular detectada (Runs idênticas). Assumindo Omega = 1.0")
            return 1.0

        # Outros erros (ex: falha de convergência numérica não relacionada a singularidade)
        print(f"❌ Erro inesperado no cálculo do Omega: {e}")
        return None


def fast_krippendorff_alpha(df: pd.DataFrame) -> float or None:
    """
    Calcula o Alpha de Krippendorff para dados de Nível INTERVALAR (numéricos)
    usando implementação vetorizada de alta performance (NumPy).

    Resolve o problema de travamento (High Cardinality) da biblioteca padrão.

    Argumentos:
        df (pd.DataFrame): DataFrame contendo os dados.
                           IMPORTANTE: O código tenta detectar automaticamente a orientação,
                           mas assume que o objetivo é medir o acordo entre AVALIADORES.
    """

    # 1. Preparação dos Dados
    # Converte para numérico e numpy array
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    matrix = df_numeric.values

    # --- DETECÇÃO E CORREÇÃO DE ORIENTAÇÃO ---
    # Krippendorff espera uma matriz (n_avaliadores, n_itens).
    # Com base no seu relato (41 avaliadores), vamos garantir essa forma.
    rows, cols = matrix.shape

    # print(f"✅ Dimensões finais para cálculo (Avaliadores x Itens): {matrix.shape}")

    # 2. Lógica Vetorizada (Krippendorff Interval Metric)
    # Referência: Krippendorff (2004), p. 230 - Fórmula simplificada para Interval

    # Mascara para ignorar NaNs
    masked_matrix = np.ma.masked_invalid(matrix)

    # Número de avaliadores por item (coluna)
    # Conta quantos não são NaN em cada coluna
    m_u = masked_matrix.count(axis=0)

    # Filtra itens que têm menos de 2 avaliações (não geram desacordo mensurável)
    valid_items_indices = np.where(m_u >= 2)[0]

    if len(valid_items_indices) < 2:
        print("⚠️ Dados insuficientes: Menos de 2 itens possuem sobreposição de avaliadores.")
        return None

    # Seleciona apenas as colunas válidas
    masked_matrix = masked_matrix[:, valid_items_indices]
    m_u = m_u[valid_items_indices]

    N = masked_matrix.count()  # Número total de pares valor-avaliador

    if N == 0:
        return None

    # --- CÁLCULO DO OBSERVED DISAGREEMENT (Do) ---
    # Do = Média das variâncias dentro de cada item (coluna)

    # Variância dentro de cada unidade (multiplicada pelos graus de liberdade apropriados)
    # Fórmula vetorizada: Soma dos quadrados das diferenças para a média da coluna

    item_means = masked_matrix.mean(axis=0)

    # Diferença de cada valor para a média do seu item
    diffs = masked_matrix - item_means

    # Soma dos quadrados das diferenças (Sum of Squares Within)
    ss_within = np.sum(diffs ** 2)

    # Ajuste para a fórmula do Krippendorff (que usa pares)
    # Do ≈ (Sum SS_within * 2) / (N * (Mean(m_u) - 1)) ?
    # Vamos usar a formula direta de soma de erros quadráticos:
    # Do = (1 / N) * sum( sum( (val_ik - val_il)^2 ) / (m_u - 1) )

    # Otimização matemática: sum((x_i - x_j)^2) = 2 * m_u * Variance * (m_u) ??
    # Identidade: Sum_{i,j} (x_i - x_j)^2 = 2 * m_u * sum(x_i - mean)^2

    numerator_parts = 2 * m_u * np.sum(diffs ** 2, axis=0)
    observed_disagreement = np.sum(numerator_parts / (m_u - 1))

    Do = observed_disagreement / N

    # --- CÁLCULO DO EXPECTED DISAGREEMENT (De) ---
    # De = Variância total de todos os dados misturados

    flat_data = masked_matrix.compressed()  # Todos os valores válidos em uma lista 1D
    total_mean = flat_data.mean()

    # Soma total dos quadrados das diferenças (Total Sum of Squares)
    ss_total = np.sum((flat_data - total_mean) ** 2)

    # De = (2 * SS_total) / (N - 1)
    De = (2 * ss_total) / (N - 1)

    # 3. Resultado Final
    if De == 0:
        print("⚠️ Variância esperada é zero (todos os valores são idênticos).")
        return 1.0 if Do == 0 else 0.0

    alpha = 1 - (Do / De)

    return float(alpha)

def calculate_krippendorff_alpha(df: pd.DataFrame) -> float or None:
    """
    Calcula o Alpha de Krippendorff para todos os dados no DataFrame de entrada.

    Argumentos:
        df (pd.DataFrame): O DataFrame de entrada. Deve ter Avaliadores (linhas)
                           e Itens/Pontuações (colunas).

    Retorna:
        float: O valor do Alpha de Krippendorff, ou None se não for calculável.
    """

    # 1. Pré-verificação de Dados
    num_items = df.shape[1]

    if df.empty or num_items < 2:
        print("⚠️ Pulando cálculo: O DataFrame está vazio ou tem menos de 2 itens (colunas).")
        return None

    # 2. Preparação e Coerção
    # Garantir que todos os dados sejam numéricos. 'coerce' transforma não-numéricos em NaN.
    numeric_data = df.apply(pd.to_numeric, errors='coerce')

    print(f"✅ Dimensões finais para cálculo (Avaliadores x Itens): {numeric_data.shape}")

    # Mantém apenas colunas (Itens) com pelo menos 2 avaliações não-nulas
    numeric_data = numeric_data.dropna(axis=1, thresh=2)

    # Se houver qualquer NaN (dado ausente), o krippendorff.alpha() lida com isso.
    if numeric_data.isnull().values.any():
        print("Aviso: Dados não numéricos ou ausentes encontrados. Serão tratados como ausentes (NaN).")

    # 3. Execução do Cálculo
    try:
        # A biblioteca Krippendorff espera a matriz (avaliadores, itens).
        # Você mencionou que seu DF já está como (41 avaliadores, N itens),
        # então usamos .values diretamente.

        # 'level_of_measurement="interval"' é mantido, ideal para pontuações numéricas.
        alpha_value = krippendorff.alpha(
            numeric_data.values,
            level_of_measurement='interval'
        )

        return alpha_value

    except ValueError as e:
        print(f"❌ Erro ao calcular Alpha de Krippendorff: {e}")
        print("Verifique se há variância suficiente nos dados ou se o número de colunas/itens é adequado.")
        return None
    except Exception as e:
        print(f"❌ Erro inesperado no cálculo do Alpha: {e}")
        return None

def calculate_cronbach_alpha(df: pd.DataFrame) -> Optional[float]:
    """
    Calcula o Alpha de Cronbach (consistência interna) para os itens no DataFrame.

    O Alpha de Cronbach é uma medida da consistência com que os itens (colunas)
    de uma escala medem um único construto unidimensional.

    Argumentos:
        df (pd.DataFrame): O DataFrame de entrada. Deve ter Avaliadores/Observações (linhas)
                           e Itens/Pontuações (colunas).

    Retorna:
        float: O valor do Alpha de Cronbach, ou None se não for calculável.
    """

    # 1. Pré-verificação de Dados
    num_items = df.shape[1]

    if df.empty or num_items < 2:
        print("⚠️ Pulando cálculo: O DataFrame está vazio ou tem menos de 2 itens (colunas).")
        return None

    # 2. Preparação e Coerção
    # Garantir que todos os dados sejam numéricos.
    numeric_data = df.apply(pd.to_numeric, errors='coerce')

    # Drop colunas onde todos os valores são NaN
    numeric_data = numeric_data.dropna(axis=1, how='all')

    print(f"✅ Dimensões finais para cálculo (Observações x Itens): {numeric_data.shape}")

    # Novo check de itens após a limpeza de colunas vazias
    if numeric_data.shape[1] < 2:
        print("⚠️ Pulando cálculo: Menos de 2 itens (colunas) válidos após a limpeza de dados.")
        return None

    # 3. Execução do Cálculo usando pingouin
    try:
        # A função cronbach_alpha do pingouin espera o DataFrame
        # A biblioteca lida com dados ausentes (NaN) ignorando a observação
        # (linha) que contém o NaN para o cálculo da variância/covariância.

        # O pingouin retorna uma série com o 'Cronbach alpha' no índice 0
        alpha_series = pg.cronbach_alpha(data=numeric_data)

        # O resultado é uma tupla ou série. Usamos o primeiro elemento que é o alpha.
        alpha_value = alpha_series[0]

        # Verifica se o resultado é um valor numérico válido (não NaN)
        if pd.isna(alpha_value):
             print("❌ O cálculo resultou em NaN. Isso pode ocorrer se a variância total dos itens for zero.")
             return None

        return float(alpha_value)

    except ValueError as e:
        print(f"❌ Erro ao calcular Alpha de Cronbach: {e}")
        print("Verifique se há variância suficiente nos dados ou se o número de observações é adequado.")
        return None
    except Exception as e:
        print(f"❌ Erro inesperado no cálculo do Alpha de Cronbach: {e}")
        return None

def calculate_xrr_metrics(df_human_ratings: pd.DataFrame,
                          df_llm_ratings_runs: pd.DataFrame,
                          llm_irr: float,
                          human_irr: float = None) -> dict:
    """
    Calcula as métricas do framework Cross-replication Reliability (xRR) entre
    um pool de avaliadores humanos (com dados ausentes) e um pool de LLM (denso).

    ATENÇÃO!!! Linhas devem ser avaliadores, colunas devem ser os items

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

        llm_ratings_item = llm_pool_df[item_name].values

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
        # print(f"Debug: d_o={d_o}, d_e={d_e}")
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


def print_result(llm, xrr_results):
    # Criação de um dicionário com a estrutura da tabela
    data = {
        "LLM": [llm],
        "Kappa_x (Norm)": [f"{xrr_results['normalized_kappa_x']:.4f}"],
        "Kappa_x (Bruto)": [f"{xrr_results['kappa_x']:.4f}"],
        "Desacordo (Do)": [f"{xrr_results['d_o']:.4f}"],
        "Desacordo (De)": [f"{xrr_results['d_e']:.4f}"],
        "IRR Humano": [f"{xrr_results['irr_human']:.4f}"],
        "IRR Modelo": [f"{xrr_results['irr_llm']:.4f}"]
    }

    # Criação do DataFrame
    df = pd.DataFrame(data)

    # Exibição (to_string index=False remove o índice numérico da esquerda)
    # print(f"\nResultados xRR para {llm}:")
    print(df.to_string(index=False))


def print_results(all_results):
    """
    Imprime uma tabela comparativa com os resultados de xRR para todos os LLMs.

    Args:
        all_results (dict): Dicionário onde a chave é o nome do LLM e o valor
                            é um dicionário com os resultados (xrr_results).
    """
    rows = []

    for llm, results in all_results.items():
        # Criação de uma linha para o LLM atual
        row = {
            "LLM": llm,
            "Kappa_x (Norm)": f"{results['normalized_kappa_x']:.4f}",
            "Kappa_x (Bruto)": f"{results['kappa_x']:.4f}",
            "Desacordo (Do)": f"{results['d_o']:.4f}",
            "Desacordo (De)": f"{results['d_e']:.4f}",
            "IRR Humano": f"{results['irr_human']:.4f}",
            "IRR Modelo": f"{results['irr_llm']:.4f}"
        }
        rows.append(row)

    # Criação do DataFrame com todas as linhas de uma vez
    df = pd.DataFrame(rows)

    # Exibição da tabela completa
    print(df.sort_values(by=['Kappa_x (Norm)'], ascending=False).to_string(index=False))

##########################################################################--------------------------------------------

df_human_scalabrino = loadScalabrinoHumanScores("Scalabrino/scores.csv")
df_scalabrino = loadScalabrino('Scalabrino/scalabrino.csv')

llms = ['Gemini25flash-lite-thinking', 'Gemini25flash-lite']# ,'DeepSeek-V3.2-Exp-thinking' , 'Gemini25flash-thinking']
llm_dfs = {}
llms_scores = {}
df_omega = {}
all_results = {}

# print(df_human_scalabrino)
# print(df_scalabrino)

alpha_scalabrino = calculate_krippendorff_alpha(df_human_scalabrino.transpose())
print(f"\nAlpha de Krippendorff calculado para Humanos de Scalabrino: {alpha_scalabrino}")
alpha_cronbach = calculate_cronbach_alpha(df_human_scalabrino.transpose())
print(f'\nAlpha de Cronbach: {alpha_cronbach}')

print("\n--- Calculando Métricas xRR (Cross-replication Reliability) ---")


for llm in llms:
    llm_dfs[llm] = loadLLMScores(llm, 'Scalabrino')
    # print(llm_dfs[llm])
    llms_scores[llm] = calculate_mean_scores(llm_dfs[llm])
    # print(llms_scores[llm])

    omega_geral = calculate_mcdonalds_omega_overall(llm_dfs[llm])
    # print(f"{llm} \t- (ω): \t\t{omega_geral:.4f}")
    df_omega[llm] = omega_geral

    # 1. Transposição Obrigatória
    # A função espera (Linhas=Runs/Avaliadores, Colunas=Itens)
    # Seus dados estão (Linhas=Itens, Colunas=Runs/Avaliadores), então usamos .T
    df_human_transposed = df_human_scalabrino.transpose()
    df_llm_transposed =  llm_dfs[llm].transpose()

    # 2. Chamada da Função
    xrr_results = calculate_xrr_metrics(
        df_human_ratings=df_human_transposed,
        df_llm_ratings_runs=df_llm_transposed,
        llm_irr=omega_geral,    # O valor 1.0 que você calculou
        human_irr=alpha_scalabrino   # O valor ~0.149 que você calculou
    )

    all_results[f'Scalabrino - {llm}'] = xrr_results
    # print_result(llm, xrr_results)


xrr_results = calculate_xrr_metrics(
        df_human_ratings=df_human_scalabrino.transpose(),
        df_llm_ratings_runs=df_scalabrino.transpose(),
        llm_irr=1,    # O valor 1.0 que você calculou
        human_irr=alpha_scalabrino   # O valor ~0.149 que você calculou
    )

all_results['Scalabrino'] = xrr_results
# print_result('Scalabrino', xrr_results)

print("Dorn")

languages = ['Cuda', 'Java', 'Python']
snippet_index = {'Cuda':0, 'Java':101, 'Python':0}
df_human = {}
alpha = {}
for language in languages:
    df_human[language] = loadHumanScores(language, snippet_index[language])
    # print(f'df_human[{language}]\n', df_human[language])
    alpha[language] = fast_krippendorff_alpha(df_human[language].transpose())
    # print(f'alpha[{language}] = ', alpha[language])

    for llm in llms:
        llm_dfs[llm] = loadLLMScores(llm, f'Dorn{language}')
        # print(llm_dfs[llm])
        llms_scores[llm] = calculate_mean_scores(llm_dfs[llm])
        # print(llms_scores[llm])

        omega_geral = calculate_mcdonalds_omega_overall(llm_dfs[llm])
        # print(f"{llm} \t- (ω): \t\t{omega_geral:.4f}")
        df_omega[llm] = omega_geral

        df_human_transposed = df_human[language].transpose()
        df_llm_transposed =  llm_dfs[llm].transpose()

        # 2. Chamada da Função
        xrr_results = calculate_xrr_metrics(
            df_human_ratings=df_human_transposed,
            df_llm_ratings_runs=df_llm_transposed,
            llm_irr=omega_geral,    # O valor 1.0 que você calculou
            human_irr=alpha[language]   # O valor ~0.149 que você calculou
        )

        all_results[f'Dorn{language} - {llm}'] = xrr_results
        # print_result(f'{llm} ({language})', xrr_results)


print_results(all_results)