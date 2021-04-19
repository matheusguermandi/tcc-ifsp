import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
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

# svm = svm.SVC()
svm = svm.SVC(random_state=0)
svm.fit(x_treino, y_treino)

# Realiza testes
print(f'Precisão: {precision_score(svm.predict(x_testes), y_testes):.2}')
print(f'F1: {f1_score(svm.predict(x_testes), y_testes):.2}')
print(f'Recall: {recall_score(svm.predict(x_testes), y_testes):.2}')

plot_confusion_matrix(svm, x_testes, y_testes)
plot_roc_curve(svm, x_testes, y_testes)
plot_precision_recall_curve(svm, x_testes, y_testes)

plt.show()
