import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')
# Certifique-se de que o estilo 'seaborn-v0_8' esteja disponível, ou use apenas 'seaborn'
# Se der erro, remova '-v0_8' ou a linha inteira.
plt.style.use('seaborn-v0_8')

# --- Carregamento do Dataset ---
# Certifique-se de que 'hotel_bookings.csv' esteja no mesmo diretório do seu script Python
# ou forneça o caminho completo para o arquivo.
df = pd.read_csv('hotel_bookings.csv')

# --- Visualização inicial, informações e estatísticas descritivas ---
print('Primeiras 5 linhas do DataFrame:')
print(df.head())
print('\nInformações do DataFrame:')
df.info() # df.info() já imprime diretamente
print('\nEstatísticas Descritivas:')
print(df.describe(include='all').T)

# --- Verificação de nulos e Duplicados ---
print('\nContagem de valores nulos por coluna:')
print(df.isnull().sum())
print('\nNúmero de linhas duplicadas:')
print(df.duplicated().sum())

# --- Removendo Duplicatas ---
df = df.drop_duplicates().reset_index(drop=True)
print('\nDataFrame após remover duplicatas e resetar o índice:')
print(df.head())

# --- Conversão de datas ---
if {'arrival_date_year','arrival_date_month','arrival_date_day_of_month'}.issubset(df.columns):
    meses = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
    'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
    df['arrival_month_num'] = df['arrival_date_month'].map(meses)
    df['data_chegada'] = pd.to_datetime({'year': df['arrival_date_year'], 'month': df['arrival_month_num'], 'day': df['arrival_date_day_of_month']})
    df = df.drop(columns=['arrival_month_num'])
print('\nDataFrame após conversão de datas e criação de data_chegada:')
print(df.head())

# --- Garantindo Tipos numéricos e Categóricos ---
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
print('\nColunas Numéricas:', num_cols)
print('Colunas Categóricas:', cat_cols)

# --- Tratamento de valores ausentes por coluna (Imputação simplificada) ---
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# Verificação pós-imputação
print('\nContagem de valores nulos após imputação:')
print(df.isnull().sum())

# --- Padronizando nomes das colunas e mudando para português ---
renomear = {
'hotel':'hotel_tipo',
'is_canceled':'cancelado',
'lead_time':'antecedencia_dias',
'arrival_date_year':'ano_chegada',
'arrival_date_month':'mes_chegada',
'arrival_date_week_number':'semana_chegada',
'arrival_date_day_of_month':'dia_chegada',
'stays_in_weekend_nights':'noites_fim_semana',
'stays_in_week_nights':'noites_semana',
'adults':'adultos',
'children':'criancas',
'babies':'bebes',
'meal':'regime',
'country':'pais',
'market_segment':'segmento_mercado',
'distribution_channel':'canal_distribuicao',
'reserved_room_type':'tipo_quarto_reservado',
'assigned_room_type':'tipo_quarto_atribuido',
'booking_changes':'alteracoes_reserva',
'deposit_type':'tipo_deposito',
'agent':'agente',
'company':'empresa',
'days_in_waiting_list':'dias_lista_espera',
'customer_type':'tipo_cliente',
'adr':'tarifa_diaria_media',
'required_car_parking_spaces':'vagas_estacionamento',
'total_of_special_requests':'pedidos_especiais'
}
df = df.rename(columns={k:v for k,v in renomear.items() if k in df.columns})
print('\nColunas do DataFrame após renomeação:')
print(df.columns)

# --- Tratando outliers ---
if 'tarifa_diaria_media' in df.columns:
    limite_adr = df['tarifa_diaria_media'].quantile(0.99)
    print('\nLimite ADR 99%:', limite_adr)
    df.loc[df['tarifa_diaria_media']>limite_adr,'tarifa_diaria_media'] = limite_adr

if 'antecedencia_dias' in df.columns:
    limite_lt = df['antecedencia_dias'].quantile(0.99)
    print('Limite de antecedência 99%:', limite_lt)
    df.loc[df['antecedencia_dias']>limite_lt,'antecedencia_dias'] = limite_lt

# --- Análise Exploratória ---

# Taxa de cancelamento por tipo de hotel
if 'hotel_tipo' in df.columns:
    resumo_hotel = df.groupby('hotel_tipo').agg(
        total_reservas = ('cancelado','count'),
        cancelamentos = ('cancelado','sum')
    ).assign(taxa_cancelamento = lambda x: x['cancelamentos']/x['total_reservas'] )
    resumo_hotel = resumo_hotel.sort_values('taxa_cancelamento', ascending=False)
    print('\nResumo por tipo de hotel:')
    print(resumo_hotel)

# Cancelamentos por país (top 10)
if 'pais' in df.columns:
    top_paises = df['pais'].value_counts().head(20).index.tolist()
    resumo_paises = df[df['pais'].isin(top_paises)].groupby('pais').agg(
        total_reservas=('cancelado','count'),
        cancelamentos=('cancelado','sum')
    ).assign(taxa_cancelamento=lambda x: x['cancelamentos']/x['total_reservas']).sort_values('total_reservas', ascending=False)
    print('\nResumo de cancelamentos por país (Top 20):')
    print(resumo_paises)

# Lead time vs taxa de cancelamento
if 'antecedencia_dias' in df.columns:
    max_val_lead_time = df['antecedencia_dias'].max()

    bins = [0, 7, 14, 30, 60, 90, 180]
    labels = ['0-7', '8-14', '15-30', '31-60', '61-90', '91-180']

    if max_val_lead_time > 180:
        if max_val_lead_time > 365:
            bins.append(365)
            labels.append('181-365')
            bins.append(max_val_lead_time) # Adiciona o valor máximo como limite superior do último bin
            labels.append(f'>365') # Ajusta o rótulo para >365
        else: # max_val_lead_time está entre 181 e 365 (inclusive)
            bins.append(max_val_lead_time) # Adiciona o valor máximo como limite superior do último bin
            labels.append(f'181-{int(max_val_lead_time)}') # Ajusta o rótulo

    # Garante que os 'bins' e 'labels' tenham o tamanho correto. bins deve ter len(labels) + 1.
    # Se max_val_lead_time for menor que o último bin existente, ajustamos.
    if len(bins) != len(labels) + 1:
        # Simplifica: se o max_val_lead_time não ultrapassar o último limite do bin, remove os bins/labels extras.
        while len(bins) > len(labels) + 1:
            bins.pop()
            labels.pop()
        if bins[-1] < max_val_lead_time:
             bins.append(max_val_lead_time)
             labels.append(f'{int(bins[-2])+1}-{int(max_val_lead_time)}')


    df['faixa_antecedencia'] = pd.cut(df['antecedencia_dias'], bins=bins, labels=labels, include_lowest=True)
    resumo_antecedencia = df.groupby('faixa_antecedencia').agg(
        total_reservas=('cancelado','count'),
        cancelamentos=('cancelado','sum')
    ).assign(taxa_cancelamento=lambda x: x['cancelamentos']/x['total_reservas'])
    print('\nResumo de cancelamentos por faixa de antecedência:')
    print(resumo_antecedencia)

# --- Visualizações - Gráficos ---

# Taxa por tipo de Hotel
plt.figure(figsize=(8,4))
sns.barplot(x=resumo_hotel.index, y=resumo_hotel['taxa_cancelamento'].values)
plt.title('Taxa de cancelamento por tipo de hotel')
plt.ylabel('Taxa de cancelamento')
plt.xlabel('Tipo de hotel')
plt.show()

# Lead time vs taxa
plt.figure(figsize=(10,4))
sns.barplot(x=resumo_antecedencia.index, y=resumo_antecedencia['taxa_cancelamento'].values)
plt.title('Taxa de cancelamento por faixa de antecedência (dias)')
plt.xlabel('Faixa de antecedência')
plt.ylabel('Taxa de cancelamento')
plt.show()

# Distribuição da tarifa diária média
if 'tarifa_diaria_media' in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df['tarifa_diaria_media'], bins=50, kde=True)
    plt.title('Distribuição da tarifa diária média (ADR)')
    plt.xlabel('Tarifa diária média')
    plt.show()

# Heatmap correlações numéricas
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlação entre variáveis numéricas')
plt.show()

# Sazonalidade: reservas e cancelamentos por mês
if 'data_chegada' in df.columns:
    df['mes_ano'] = df['data_chegada'].dt.to_period('M').astype(str)
    serie_mensal = df.groupby('mes_ano').agg(
    total_reservas=('cancelado','count'),
    cancelamentos=('cancelado','sum')
    ).assign(taxa_cancelamento=lambda x: x['cancelamentos']/x['total_reservas'])
    serie_mensal[['total_reservas','cancelamentos']].plot(figsize=(12,4))
    plt.title('Reservas e cancelamentos ao longo do tempo (mensal)')
    plt.show()
    serie_mensal['taxa_cancelamento'].plot(figsize=(12,3))
    plt.title('Taxa de cancelamento mensal')
    plt.show()

# Análise da Tarifa Diária Média (ADR) vs. Cancelamento
if 'tarifa_diaria_media' in df.columns and 'cancelado' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='cancelado', y='tarifa_diaria_media', data=df, palette='viridis')
    plt.title('Tarifa Diária Média (ADR) por Status de Cancelamento')
    plt.xlabel('Cancelado (0: Não Cancelado, 1: Cancelado)')
    plt.ylabel('Tarifa Diária Média (ADR)')
    plt.xticks([0, 1], ['Não Cancelado', 'Cancelado']) # Melhora os rótulos do eixo X
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# --- Features e modelagem preditiva ---
features = []
for c in ['hotel_tipo','antecedencia_dias','noites_fim_semana','noites_semana','adultos','criancas','bebes','regime','pais','segmento_mercado','canal_distribuicao','tipo_quarto_reservado','tipo_cliente','tarifa_diaria_media','pedidos_especiais']:
    if c in df.columns:
        features.append(c)

X = df[features].copy()
y = df['cancelado'].astype(int)

# Tratando valores categóricos e numéricos separadamente
cat_feats = X.select_dtypes(include=['object','category']).columns.tolist()
num_feats = X.select_dtypes(include=['int64','float64']).columns.tolist()

# Pipeline para pré-processamento
preprocessor = ColumnTransformer(transformers=[
('num', StandardScaler(), num_feats),
('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
])

# Divisão treino/teste 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Treinando Random Forest com pipeline
pipeline = Pipeline(steps=[
('preproc', preprocessor),
('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])
pipeline.fit(X_train, y_train)

# --- Avaliação do modelo ---
y_pred = pipeline.predict(X_test)
print('\nAcurácia:', accuracy_score(y_test, y_pred))
print('\nMatriz de confusão:')
print(confusion_matrix(y_test, y_pred))
print('\nRelatório de classificação:')
print(classification_report(y_test, y_pred))

# Visualização da Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Não Cancelado', 'Cancelado'],
            yticklabels=['Não Cancelado', 'Cancelado'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

# --- Extrair importâncias do RandomForest e mostrar top 20 features ---
nomes_full = pipeline.named_steps['preproc'].get_feature_names_out().tolist()
importancias = pipeline.named_steps['clf'].feature_importances_
imp_df = pd.DataFrame({'feature':nomes_full,'importancia':importancias}).sort_values('importancia',ascending=False).head(20)
print('\nImportância das Features (Top 20):')
print(imp_df)