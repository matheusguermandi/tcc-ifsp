import numpy as np
import paraconsistent
from load_data import load_csv

base_path = 'C:/Users/guermandi/Desktop/TCC/AnaliseParaconsistente/tests'

pathological = load_csv(f'{base_path}/patologicos-normalizados.csv')
healthy = load_csv(f'{base_path}/saudaveis-normalizados.csv')

pathological = np.delete(pathological, (0, 1, 2, 4, 5, 6), 1)
healthy = np.delete(healthy, (0, 1, 2, 4, 5, 6), 1)

classes = np.array([np.array(pathological), np.array(healthy)])

alpha = paraconsistent.alpha(classes)
beta = paraconsistent.beta(classes)

assurance = paraconsistent.assurance(alpha, beta)
contradiction = paraconsistent.contradiction(alpha, beta)

truth = paraconsistent.truth(assurance, contradiction)

# Classes de dados
classes = np.array([np.array(pathological), np.array(healthy)])

# Alfa e Beta
alpha = paraconsistent.alpha(classes)
beta = paraconsistent.beta(classes)

# Ponto G1
assurance = paraconsistent.assurance(alpha, beta)

# Ponto G2
contradiction = paraconsistent.contradiction(alpha, beta)

falsehood = paraconsistent.falsehood(assurance, contradiction)
truth = paraconsistent.truth(assurance, contradiction)
indefinition = paraconsistent.indefinition(assurance, contradiction)
ambiguity = paraconsistent.ambiguity(assurance, contradiction)

print(f'Alfa: {alpha:.3f}')
print(f'Beta: {beta:.3f}\n')

print(f'P=(G1, G2) = ({assurance:.3f}, {contradiction:.3f})\n')

print(f'Distância de P para o ponto (-1, 0) = {falsehood:.3f}')
print(f'Distância de P para o ponto (1, 0) = {truth:.3f}')
print(f'Distância de P para o ponto (0, -1) = {indefinition:.3f}')
print(f'Distância de P para o ponto (0, 1) = {ambiguity:.3f}')
