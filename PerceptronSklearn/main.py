import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix, plot_precision_recall_curve, precision_score, \
    f1_score, recall_score
from sklearn.model_selection import train_test_split

# Carrega dataset
# dados = pd.read_csv('dataset-selecionado.csv')
dados = pd.read_csv('dataset-todos.csv')
dados = np.array(dados)

# Separa dados de treinamento e testes
tamanho_conjunto = 14

X = np.delete(np.array(dados), tamanho_conjunto, 1)
Y = np.array(dados)[:, tamanho_conjunto]

x_treino, x_testes, y_treino, y_testes = train_test_split(X, Y, test_size=0.3, shuffle=False)

# Realiza treinamento da rede
pmc = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(3,), activation='tanh', random_state=1, max_iter=1000)
pmc.fit(x_treino, y_treino)

# Realiza testes
print(f'Precisão: {precision_score(pmc.predict(x_testes), y_testes):.2}')
print(f'F1: {f1_score(pmc.predict(x_testes), y_testes):.2}')
print(f'Recall: {recall_score(pmc.predict(x_testes), y_testes):.2}')

plot_confusion_matrix(pmc, x_testes, y_testes)
plot_roc_curve(pmc, x_testes, y_testes)
plot_precision_recall_curve(pmc, x_testes, y_testes)

plt.show()