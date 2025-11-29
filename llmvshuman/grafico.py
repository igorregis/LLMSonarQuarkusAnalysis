import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Tenta importar adjustText para evitar sobreposição de labels
# Se não tiver instalado, use: pip install adjustText
try:
    from adjustText import adjust_text
    has_adjust_text = True
except ImportError:
    has_adjust_text = False

# --- DADOS COMPLETOS EXTRAÍDOS DO TEXTO FORNECIDO ---
data = {
    'Model': [
        'Grok-4-1-fast-thinking', 'Claude45-sonnet', 'Grok-4-0709-thinking', 'Claude45-opus', 'Claude45-opus-thinking',
        'Kimi-K2-thinking', 'Gemini20pro', 'Gemini25flash-lite',
        'DeepSeek-V3.2-Exp-thinking', 'DeepSeek-V3.2-Exp', 'Qwen3-235B-thinking',
        'Gemini25flash-lite-thinking-10k', 'Claude37-sonnet', 'Claude45-sonnet-thinking',
        'Claude45-haiku-thinking', 'Gemini25flash-lite-thinking-1k', 'GPT-oss-120b-thinking',
        'GPT-5-thinking', 'Gemini30pro', 'Gemini25flash-thinking',
        'Claude45-haiku', 'Gemini25flash', 'Llama31-405b',
        'Gemini25pro', 'Llama-4-Maverick-17B', 'GPT-51-thinking',
        'Qwen3-Coder-480B', 'Claude35-haiku', 'GPT-5-mini-thinking',
        'DeepSeek-V3', 'GPT-5', 'GPT-5-nano-thinking',
        'GPT4o', 'Llama31-8b', 'GPT4o-mini',
        'Gemini20flash', 'Llama-4-Scout-17B', 'GPT-5-mini',
        'SCO (Baseline)', 'GPT-5-nano'
    ],
    'Family': [
        'Grok', 'Claude', 'Grok', 'Claude', 'Claude',
        'Kimi', 'Gemini', 'Gemini',
        'DeepSeek', 'DeepSeek', 'Qwen',
        'Gemini', 'Claude', 'Claude',
        'Claude', 'Gemini', 'GPT',
        'GPT', 'Gemini', 'Gemini',
        'Claude', 'Gemini', 'Llama',
        'Gemini', 'Llama', 'GPT',
        'Qwen', 'Claude', 'GPT',
        'DeepSeek', 'GPT', 'GPT',
        'GPT', 'Llama', 'GPT',
        'Gemini', 'Llama', 'GPT',
        'Scalabrino', 'GPT'
    ],
    'xRR_Norm': [
        0.8300, 0.8207, 0.8016, 0.8273, 0.8210,
        0.7908, 0.7903, 0.7868,
        0.7859, 0.7859, 0.7859,
        0.7765, 0.7684, 0.7578,
        0.7519, 0.7511, 0.7444,
        0.7428, 0.7405, 0.7400,
        0.7331, 0.7202, 0.7137,
        0.7034, 0.6892, 0.6786,
        0.6753, 0.6730, 0.6720,
        0.6028, 0.5965, 0.5959,
        0.5631, 0.5516, 0.5422,
        0.5218, 0.5067, 0.4850,
        0.4423, 0.3266
    ],
    'Macro_F1': [
        0.66, 0.64, 0.51, 0.65, 0.68,
        0.63, 0.56, 0.62,
        0.74, 0.70, 0.69,
        0.69, 0.70, 0.69,
        0.64, 0.61, 0.64,
        0.59, 0.59, 0.55,
        0.55, 0.61, 0.48,
        0.58, 0.52, 0.54,
        0.57, 0.50, 0.65,
        0.36, 0.53, 0.62,
        0.32, 0.48, 0.32,
        0.43, 0.37, 0.42,
        0.49, 0.34
    ]
}

df = pd.DataFrame(data)

# --- CONFIGURAÇÃO VISUAL ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(18, 10)) # Tamanho aumentado para comportar os dados

# Paleta de cores distinta para as famílias
palette = sns.color_palette("bright", n_colors=len(df['Family'].unique()))

# Plotagem
scatter = sns.scatterplot(
    data=df,
    x='Macro_F1',
    y='xRR_Norm',
    hue='Family',
    style='Family',
    s=150,
    palette=palette,
    alpha=0.85,
    edgecolor='black' # Borda preta para destacar os pontos
)

# --- ANOTAÇÕES DOS PONTOS ---
texts = []
for i in range(df.shape[0]):
    # Adiciona o nome do modelo.
    # Reduzimos a fonte ligeiramente para 8pt devido à quantidade de itens
    texts.append(plt.text(df.Macro_F1[i], df.xRR_Norm[i], df.Model[i], fontsize=8, weight='bold'))

# Ajuste automático de labels para não sobrepor
if has_adjust_text:
    adjust_text(
        texts,
        arrowprops=dict(
            arrowstyle='-',
            color='gray',
            lw=0.5,
            shrinkA=15,
        ),
        expand_points=(1.5, 1.2)
    )
else:
    print("Aviso: Biblioteca 'adjustText' não encontrada. Labels podem se sobrepor.")

# --- LINHAS DE REFERÊNCIA E DECORAÇÃO ---

# Linha Horizontal: Corte de "Forte Concordância" do xRR (ex: 0.8 ou 0.75 dependendo da literatura, aqui visual)
plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, label='Alta Concordância Humana (xRR > 0.8)')

# Linha Vertical: F1 Score de referência (ex: humano principal)
plt.axvline(x=0.67, color='blue', linestyle='--', alpha=0.3, label='F1 Humano (Ref: Principal)')

# Títulos e Eixos
# plt.title('Panorama Geral: Acurácia (F1) vs Alinhamento Humano (xRR)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Macro F1-Score (Acurácia Multiclasse)', fontsize=12)
plt.ylabel('xRR Normalizado (Similaridade de Julgamento)', fontsize=12)

# Legenda
# plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Família', frameon=True)
plt.legend(
    loc='lower right', # Coloca o canto inferior direito da legenda no ponto de ancoragem
    bbox_to_anchor=(1, 0), # Ancoragem no canto inferior direito da área de plotagem
    title='Família',
    frameon=True # Mantém o frame da legenda para melhor contraste
)

plt.grid(True, linestyle=':', alpha=0.5)

# Salvar e Mostrar
plt.tight_layout()
plt.savefig('graficos/scatter_full_models.png', dpi=300)
plt.show()