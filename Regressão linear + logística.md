Exercício: Regressão linear + logística
Parte 1: Regressão linear (prever preço de casas)

Dataset: California Housing, já vem no scikit-learn, sem download:

python
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing(as_frame=True).frame  # alvo: MedHouseVal

Tarefas:

Faça a análise exploratória: df.describe(), histogramas, mapa de correlação (seaborn.heatmap).
Separe treino/teste (80/20).
Treine uma LinearRegression e calcule MAE, RMSE e R² no teste.
Analise os resíduos: plote resíduos vs. previsões e verifique se há padrão (viola pressupostos?).
Compare com Ridge e Lasso usando validação cruzada (cross_val_score).
Interprete: quais 3 variáveis mais influenciam o preço? (Padronize antes para comparar coeficientes.)
Parte 2: Regressão logística (prever sobrevivência no Titanic)

Dataset: Titanic (Kaggle ou seaborn.load_dataset("titanic")).

Tarefas:

Trate nulos (age, embarked) e converta categóricas com pd.get_dummies.
Separe treino/teste com stratify=y.
Padronize e treine a LogisticRegression.
Avalie: matriz de confusão, precision, recall, F1 e AUC.
Calcule o odds ratio (np.exp(clf.coef_)) e interprete. Ex.: "ser mulher multiplica as chances de sobrevivência por X".
Varie o corte (0,3 / 0,5 / 0,7) e observe o trade-off entre precision e recall.
Entregável (para o portfólio)

Um notebook no GitHub com:

Resultados e gráficos de cada etapa.
Um parágrafo final respondendo: "Qual modelo/corte eu escolheria e por quê, em termos de negócio?"
