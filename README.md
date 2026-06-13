# Predição de Aprovação de Acesso a Recursos

Este repositório contém o desenvolvimento de um modelo de Machine Learning focado em Classificação Binária Supervisionada para prever se a solicitação de acesso de um funcionário a um determinado recurso deve ser **concedida** ou **negada**.

O principal desafio técnico deste projeto é lidar com o **severo desbalanceamento de classes**, garantindo que o modelo seja robusto, justo e capaz de identificar corretamente as tentativas de acesso negadas.

---

## 🎯 Objetivos do Projeto

- **Análise Exploratória (EDA):** Identificar o comportamento das variáveis, distribuições de frequência, cardinalidade e presença de outliers.
- **Tratamento de Dados:** Investigar a presença de dados faltantes e aplicar estratégias de imputação adequadas para dados categóricos anônimos.
- **Engenharia de Balanceamento:** Aplicar e comparar técnicas de reamostragem (como SMOTE/SMOTENC e Random Undersampling) para mitigar o viés da classe majoritária.
- **Avaliação Rigorosa:** Comparar modelos com e sem balanceamento utilizando métricas críticas para cenários desbalanceados (Precisão, Recall, F1-Score e AUC-ROC).

---

## 📊 Base de Dados

O modelo é treinado e validado utilizando a base de dados histórica do **Amazon.com - Employee Access Challenge** (disponível no Kaggle). Este dataset simula um cenário real de governança de identidade e acesso corporativo.

---

## 🛠️ Tecnologias e Ferramentas

---

## 🚀 Como Executar
