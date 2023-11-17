# Roteiro de Aula - Gráficos  com Matplotlib em Python

## 1. Introdução
- Breve explicação sobre a importância dos gráficos na visualização de dados.
- Apresentação da biblioteca `matplotlib` e sua versatilidade na criação de gráficos.

## 2. Instalação da biblioteca
- Certifique-se de que a biblioteca `matplotlib` está instalada:
  ```bash
  pip install matplotlib
  
## 3. Gráfico de Barras
- Importar o módulo pyplot do matplotlib:
```
import matplotlib.pyplot as plt
```
- Criar dados para o gráfico de barras:

```
categorias = ['Categoria A', 'Categoria B', 'Categoria C', 'Categoria D']
valores = [25, 40, 30, 45]
```
- Criar o gráfico de barras:
```
plt.bar(categorias, valores, color='blue')
plt.xlabel('Categorias')
plt.ylabel('Valores')
plt.title('Gráfico de Barras')
plt.show()
```

## 4. Gráfico de Colunas
```
# Utilizar os mesmos dados anteriores

# Criar o gráfico de colunas
plt.barh(categorias, valores, color='green')
plt.xlabel('Valores')
plt.ylabel('Categorias')
plt.title('Gráfico de Colunas')
plt.show()
```

## 5. Gráfico de Pizza

```
# Criar dados para o gráfico de pizza
labels = ['Categoria A', 'Categoria B', 'Categoria C', 'Categoria D']
sizes = [25, 40, 30, 45]

# Criar o gráfico de pizza
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'yellow', 'blue', 'green'])
plt.title('Gráfico de Pizza')
plt.show()
```

## 6. Gráfico de Dispersão (Scatter Plot)
```
import matplotlib.pyplot as plt
import numpy as np

# Criar dados para o gráfico de dispersão
x = np.random.rand(50)
y = np.random.rand(50)

# Criar o gráfico de dispersão
plt.scatter(x, y, color='purple')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.title('Gráfico de Dispersão')
plt.show()
```
##  7. Gráfico de Linhas
```
import matplotlib.pyplot as plt
import numpy as np

# Criar dados para o gráfico de linhas
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Criar o gráfico de linhas
plt.plot(x, y, color='orange')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.title('Gráfico de Linhas')
plt.show()
```
## 8. Histograma
```
import matplotlib.pyplot as plt
import numpy as np

# Criar dados para o histograma
dados = np.random.randn(1000)

# Criar o histograma
plt.hist(dados, bins=30, color='green', alpha=0.7)
plt.xlabel('Valores')
plt.ylabel('Frequência')
plt.title('Histograma')
plt.show()
```

## 9. Gráfico de Área Empilhada
```
import matplotlib.pyplot as plt

# Criar dados para o gráfico de área empilhada
categorias = ['A', 'B', 'C', 'D']
valores1 = [4, 7, 2, 5]
valores2 = [2, 6, 3, 8]

# Criar o gráfico de área empilhada
plt.stackplot(categorias, valores1, valores2, labels=['Grupo 1', 'Grupo 2'], colors=['orange', 'blue'])
plt.xlabel('Categorias')
plt.ylabel('Valores')
plt.title('Gráfico de Área Empilhada')
plt.legend()
plt.show()
```

## 10. Gráfico de Radar
```
import matplotlib.pyplot as plt
import numpy as np

# Criar dados para o gráfico de radar
categorias = ['A', 'B', 'C', 'D', 'E']
valores = [4, 7, 2, 5, 8]

# Criar o gráfico de radar
angulos = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False)
valores += valores[:1]
angulos += angulos[:1]
plt.polar(angulos, valores, marker='o', linestyle='-', color='purple')
plt.fill(angulos, valores, color='purple', alpha=0.25)
plt.title('Gráfico de Radar')
plt.show()

```

## 11. Gráfico de Caixas (Boxplot)
```
import matplotlib.pyplot as plt
import numpy as np

# Criar dados para o boxplot
dados = np.random.randn(100)

# Criar o boxplot
plt.boxplot(dados)
plt.ylabel('Valores')
plt.title('Gráfico de Caixas (Boxplot)')
plt.show()
```
## 12. Gráfico de Violino
```
import matplotlib.pyplot as plt
import seaborn as sns

# Criar dados para o gráfico de violino
dados = sns.load_dataset('iris')

# Criar o gráfico de violino
sns.violinplot(x='species', y='sepal_length', data=dados)
plt.xlabel('Espécies')
plt.ylabel('Comprimento da Sépala')
plt.title('Gráfico de Violino')
plt.show()

```
## 13. Gráfico de Mapa de Calor
```
import matplotlib.pyplot as plt
import seaborn as sns

# Criar dados para o mapa de calor
dados = sns.load_dataset('flights').pivot_table(index='month', columns='year', values='passengers')

# Criar o mapa de calor
sns.heatmap(dados, cmap='YlGnBu', annot=True, fmt='d')
plt.xlabel('Ano')
plt.ylabel('Mês')
plt.title('Mapa de Calor')
plt.show()

```
## 14. Gráfico 3D de Barras
```
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Criar dados para o gráfico 3D de barras
x = np.arange(4)
y = np.arange(4)
x, y = np.meshgrid(x, y)
z = np.array([[5, 10, 15, 20],
              [10, 15, 20, 25],
              [15, 20, 25, 30],
              [20, 25, 30, 35]])

# Criar o gráfico 3D de barras
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z).ravel(), 0.8, 0.8, z.ravel(), shade=True)
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_zlabel('Eixo Z')
ax.set_title('Gráfico 3D de Barras')
plt.show()

```
## 15. Gráfico de Waffle
```
import matplotlib.pyplot as plt
from pywaffle import Waffle

# Criar dados para o gráfico de waffle
dados = {'Categoria A': 15, 'Categoria B': 30, 'Categoria C': 45}

# Criar o gráfico de waffle
fig = plt.figure(
    FigureClass=Waffle,
    rows=5,
    columns=10,
    values=dados,
    title={'label': 'Gráfico de Waffle', 'loc': 'left'},
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(dados), 'framealpha': 0}
)
plt.show()

```
# Como salvar em imagem o gráfico?
```
plt.savefig('nome_do_arquivo.png')
```
