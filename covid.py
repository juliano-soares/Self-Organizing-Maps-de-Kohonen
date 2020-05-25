# importação das bibliotecas usadas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import scale

# importação dos dados
dataset = pd.read_csv('datasets/covidRegioesBrasil.csv')
#dataset = pd.read_csv('datasets/covidEstadosBrasil.csv')
#dataset = pd.read_csv('datasets/insumos.csv')
#dataset = pd.read_csv('datasets/covidMundo.csv')
    
# nomes das colunas para avaliação
#columns_name = ['casos','mortos', 'incidencia/100mil', 'mortalidade/100mil']
columns_name = ['casos', 'mortos', 'incidencia/100mil', 'mortalidade/100mil']
#columns_name = ["Vacinas distribuidas - influenza","Vacinas aplicadas - influenza ","Mascara cirúrgica","Mascara N95","Alcool em gel - L","Avental","Teste rápido","Luvas","Óculos e protetor facial","Touca e sapatilha","Cloroquina - comprimidos","Oseltamivir - cápsulas","Teste PCR","Leitos locados","Leitos UTI adulto","Respiradores distribuidos","UTI adulto SUS","Uti adulto não SUS","Leitos UTI habilitados","Mais Médicos"]
#columns_name = ['casos', 'mortos', 'recuperados']

# seta os nomes das colunas
X = dataset[columns_name].values
X = scale(X)

# tamanho que a tabela vai ser gerada
size = 15
# Parametros
# 1º - Dimensões da matrix eixo X
# 2º - Dimensões da matrix eixo Y
# 3º - numero de atributos
# 4º - o raio do nó vencedor
# 5º - parametro para treinamento
som = MiniSom(size, size, len(X[0]), sigma=1.5, random_seed=1)

# inicia os pesos 
som.pca_weights_init(X)
# numero de treinos
som.train_random(X, 1000)

# plotagem com gráfico 
# grafico de posição
country_map = som.labels_map(X, dataset.iloc[:, 0])
plt.figure(figsize=(15, 15))
for p, tipo in country_map.items():
    tipo = list(tipo)
    x = p[0] + .1
    y = p[1] - .3
    for i, c in enumerate(tipo):
        off_set = (i+1)/len(tipo) - 0.05
        plt.text(x, y+off_set, c, fontsize=10)
        
plt.pcolor(som.distance_map().T, cmap='binary', alpha=.2)
plt.xticks(np.arange(size+1))
plt.yticks(np.arange(size+1))
plt.grid()
plt.show()

# grafico de agrupamentos
W = som.get_weights()
plt.figure(figsize=(14, 14))
for i, tipo in enumerate(columns_name):
    if len(columns_name) > 9: # usado para o dataset insumos.csv
        plt.subplot(5, 5, i+1)    
    else:
        plt.subplot(3, 3, i+1) 
    plt.title(tipo)
    plt.pcolor(W[:,:,i].T, cmap='hsv')
    plt.xticks(np.arange(size+1))
    plt.yticks(np.arange(size+1))
plt.tight_layout()
plt.show()
