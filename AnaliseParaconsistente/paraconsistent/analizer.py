import numpy as np
from math import sqrt, pow


def range_vectors(classes):
    """
    Considerando que uma determinada classe seja uma matriz numérica, encontra os maiores e menores valores de cada
     coluna.

    :param classes: Classes de dados normalizadas.
    :return: Vetores de alcance de cada classe (maiores e menores valores de cada coluna da matriz).
    """

    largest_and_smallest = []

    for data_class in classes:
        smallest = np.array([])
        largest = np.array([])

        for array in np.transpose(data_class):
            largest = np.append(largest, np.max(array))
            smallest = np.append(smallest, np.min(array))

        largest_and_smallest.append(np.array([smallest, largest]))

    return np.array(largest_and_smallest)


def similarity_vectors(range_vectors_local):
    """
    A partir dos vetores de alcance passados por parâmetro, encontra os vetores de similaridades de cada classe.

    :param range_vectors_local: Vetores de alcance.
    :return: Vetores de similaridades (Valor 1 subraído pelo maior e menor valor dos vetores de similaridades).
    """

    vectors = []

    for values in range_vectors_local:
        similarity_vector = []

        for largest_and_smallest in np.transpose(values):
            amplitude = largest_and_smallest[1] - largest_and_smallest[0]
            similarity_vector.append(1 - amplitude)

        vectors.append(similarity_vector)

    return np.array(vectors)


def alpha(classes):
    """
    Quantifica as semelhanças intraclasse, encontrado o elemento alfa, que expressa o nível de fé nas características.

    :param classes: Classes de dados normalizadas.
    :return: Elemento alfa (menor valor entre as médias dos vetores de similaridade).
    """

    largest_and_smallest = range_vectors(classes)
    vectors = similarity_vectors(largest_and_smallest)
    means = [np.mean(similarity_vector) for similarity_vector in vectors]

    return np.min(means)


def max_overlaps(classes):
    """
    Encontra o número máximo de sobreposições (F) que uma classe pode ter. F = N * (N - 1) * X * T

    :param classes: Classes de dados normalizadas.
    :return: Quantidade máxima de sobreposições possíveis de todas as classes.
    """

    number_classes = len(classes)  # N
    number_vetors_per_class = len(classes[0])  # X
    number_elements_per_vector = len(classes[0][0])  # T

    return number_classes * (number_classes - 1) * number_vetors_per_class * number_elements_per_vector  # F


def number_overlaps(classes):
    """
    Conta o número de sobreposições numa determinada classe, percorrendo todas as classes e verificando se cada elemento
     dos vetores da classe em questão está dentro do intervalo dos vetores de alcande das outras classes.

    :param classes: Classes de dados normalizadas.
    :return: Quantidade de sobreposições de todas as classes.
    """

    range_vectors_local = range_vectors(classes)
    overlap_counter = 0

    for i_class in range(0, len(classes)):
        for vector in classes[i_class]:
            for i_vector in range(0, len(vector)):
                for i_range in range(0, len(range_vectors_local)):
                    if i_class != i_range:
                        ranges = range_vectors_local[i_range].transpose()[i_vector]

                        if ranges[0] < vector[i_vector] < ranges[1]:
                            overlap_counter += 1

    return overlap_counter


def beta(classes):
    """
    Quantifica as semelhanças interclasse, encontrado o elemento beta, que expressa o nível de descrença nas
     características.

    :param classes: Classes de dados normalizadas.
    :return: Elemento beta (divisão entre a quantidade exata e a quantidade máxima de sobreposições).
    """
    r = number_overlaps(classes)
    f = max_overlaps(classes)

    return r / f


def assurance(a, b):
    """
    Encontra o ponto G1, a partir dos elementos alfa (a) e beta (b), que expressa o grau de confiabilidade do conjunto
     de classes.

    :param a: Elemento alfa.
    :param b: Elemento beta.
    :return: Ponto G1.
    """
    return a - b


def contradiction(a, b):
    """
    Encontra o ponto G2, a partir dos elementos alfa (a) e beta (b), que expressa o grau de contradição do conjunto de
     classes.

    :param a: Elemento alfa.
    :param b: Elemento beta.
    :return: Ponto G2.
    """
    return a + b - 1


def falsehood(g1, g2):
    """
    Encontra a distância de P para o ponto (-1, 0) do plano paraconsistente, que demonstra o grau de descrédito
     (falsidade) dos dados. (G1, G2) = (-1, 0)

    :param g1: Ponto G1.
    :param g2: Ponto G2.
    :return: Nível de falsidade das classes.
    """
    return sqrt(pow(g1 + 1, 2) + pow(g2, 2))


def truth(g1, g2):
    """
    Encontra a distância de P para o ponto (1, 0) do plano paraconsistente, que demonstra o grau de veracidade dos 
     dados. (G1, G2) = (1, 0)
    
    :param g1: Ponto G1.
    :param g2: Ponto G2.
    :return: Nível de veracidade das classes.
    """
    return sqrt(pow(g1 - 1, 2) + pow(g2, 2))


def indefinition(g1, g2):
    """
    Encontra a distância de P para o ponto (0, -1) do plano paraconsistente, que demonstra o grau de indefinição dos 
     dados. (G1, G2) = (0, -1)
    
    :param g1: Ponto G1.
    :param g2: Ponto G2.
    :return: Nível de indefinição das classes.
    """
    return sqrt(pow(g1, 2) + pow(g2 + 1, 2))


def ambiguity(g1, g2):
    """
    Encontra a distância de P para o ponto (0, 1) do plano paraconsistente, que demonstra o grau de ambiguidade dos
     dados. (G1, G2) = (0, 1)
    
    :param g1: Ponto G1.
    :param g2: Ponto G2.
    :return: Nível de ambiguidade das classes.
    """
    return sqrt(pow(g1, 2) + pow(g2 - 1, 2))
