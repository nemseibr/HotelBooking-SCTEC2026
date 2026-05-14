# 📊 Previsão de Cancelamentos de Reservas Hoteleiras

Projeto desenvolvido por **Otávio Augusto Reis Nascimento** com foco na previsão de cancelamentos de reservas hoteleiras.  
O objetivo foi analisar o dataset **hotel_bookings.csv** para identificar padrões e construir um modelo preditivo eficaz.

---

## 🚀 Etapas de Desenvolvimento

O projeto seguiu uma metodologia padrão de Machine Learning, abrangendo as seguintes fases:

- **Carga e Inspeção Inicial dos Dados**  
- **Pré-processamento e Tratamento dos Dados**  
- **Análise Exploratória de Dados (EDA)**  
- **Modelagem Preditiva**  
- **Análise de Importância de Features**  
- **Conclusão e Insights**  

---

## 🛠️ Principais Decisões no Tratamento dos Dados

- Remoção de duplicatas (31.994 linhas)  
- Engenharia de features de data (`data_chegada`, `mes_ano`)  
- Tratamento de valores ausentes (mediana para numéricos, moda para categóricos)  
- Renomeação de colunas para português  
- Tratamento de outliers com capping no 99º percentil  

---

## 🔎 Principais Insights da EDA

- **City Hotels** apresentaram maior taxa de cancelamento que **Resort Hotels**  
- Países como **Portugal (PRT)** e **China (CHN)** tiveram maiores taxas de cancelamento  
- Maior antecedência da reserva → maior probabilidade de cancelamento  
- Padrões sazonais ao longo dos meses  
- Diferenças na distribuição da **tarifa média (ADR)** entre reservas canceladas e não canceladas  

---

## 🤖 Resultados do Modelo Preditivo

- **Modelo**: `RandomForestClassifier` em Pipeline com `StandardScaler` e `OneHotEncoder`  
- **Acurácia**: ~80.59% no conjunto de teste  
- **Recall classe 1 (cancelamentos)**: ~55%  
- **Features mais relevantes**: `antecedencia_dias`, `tarifa_diaria_media`, `noites_semana`, `pedidos_especiais`  

---

## 📌 Recomendações Futuras

- Aplicar técnicas de rebalanceamento de classes (ex.: **SMOTE**)  
- Ajuste fino de hiperparâmetros  
- Testar outros algoritmos de classificação  

---

## ⚙️ Como Executar o Projeto

### 1. Google Colab (Notebook Interativo)
- Upload do arquivo `Projeto_hotel_bookings.ipynb`  
- Upload do dataset `hotel_bookings.csv`  
- Executar todas as células (`Runtime > Run all`)  

### 2. VS Code (Script Python)
- Instalar extensão Python oficial da Microsoft  
- Colocar `hotel_booking_analysis.py` e `hotel_bookings.csv` na mesma pasta  
- Criar ambiente virtual e instalar dependências:  
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  
## 💻 Tecnologias Utilizadas
Python 3.x

Pandas → manipulação e limpeza de dados

NumPy → operações matemáticas e vetoriais

Matplotlib & Seaborn → visualizações gráficas

Scikit-learn → pré-processamento e modelagem preditiva

Google Colab → execução interativa em notebook

VS Code → desenvolvimento em script Python

---

## 👨‍💻 Autor
Otávio Augusto Reis Nascimento  
Projeto acadêmico com foco em Machine Learning aplicado à hotelaria. Desafio proposto por SCTEC-Florianópolis 2026
