# Predição de Aprovação de Acesso a Recursos

Este repositório contém o desenvolvimento de um modelo de Machine Learning focado em Classificação Binária Supervisionada para prever se a solicitação de acesso de um funcionário a um determinado recurso deve ser **concedida** ou **negada**.

O principal desafio técnico deste projeto é lidar com o **severo desbalanceamento de classes** (94% das solicitações são historicamente concedidas), garantindo que o modelo seja robusto, justo e capaz de identificar corretamente as tentativas de acesso negadas.

---

## 🎯 Objetivos do Projeto

- **Análise Exploratória (EDA):** Identificar o comportamento das variáveis, distribuições de frequência, alta cardinalidade de IDs e presença de anomalias por meio de análises bivariadas e correlação V de Cramer.
- **Tratamento de Dados:** Mapeamento de dados faltantes e tratamento de alta cardinalidade de recursos raros por meio de agrupamentos por frequência mínima no pré-processamento.
- **Engenharia de Balanceamento:** Investigação do impacto real da técnica de reamostragem sintética **SMOTE** no treinamento do modelo, comparando-o diretamente contra o cenário desbalanceado puro.
- **Avaliação Rigorosa:** Validação cruzada (Stratified CV) e teste final comparando múltiplos algoritmos por meio de métricas críticas para cenários desbalanceados (Acurácia, Precisão, Recall, F1-Score e AUC-ROC).

---

## 📊 Base de Dados

O modelo é treinado e validado utilizando a base de dados histórica do **Amazon.com - Employee Access Challenge** (disponível no Kaggle). Este dataset simula um cenário real de governança de identidade e acesso corporativo (IAM).

---

## 🛠️ Tecnologias e Ferramentas

- **Linguagem:** Python
- **Manipulação e Análise:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`LogisticRegression`, `RandomForestClassifier`), XGBoost (`XGBClassifier`)
- **Visualização de Dados:** Matplotlib, Seaborn
- **Persistência de Modelos:** Joblib

---

## 🚀 Como Executar

O projeto está estruturado de forma modular e sequencial em cadernos Jupyter:

1. **`01_EDA.ipynb` (Análise Exploratória):** - Execute para visualizar a distribuição das classes, o Heatmap de associação categórica e os gráficos bivariados que cruzam o comportamento do Top 10 de Gerentes e Recursos contra a decisão final de acesso.
2. **`02_Treinamento_e_Avaliacao_Modelos.ipynb` (Modelagem):** - Execute para rodar a pipeline de One-Hot Encoding, o balanceamento por SMOTE, a busca aleatória de hiperparâmetros (`RandomizedSearchCV`), e a geração das curvas ROC, matrizes de confusão e o gráfico de importância de variáveis.

---
