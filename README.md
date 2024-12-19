# Tutorial: Implementando Regras como Axiomas com LTN

Equipe:

- Elian Souza
- Josival Salvador
- Richard Auzier
- Vinícius Fonseca

Este tutorial fornece um guia para criar e testar axiomas usando abordagens inspiradas na estrutura do LTNtorch. Vamos explorar como definir, implementar e validar axiomas em um ambiente similar, destacando os princípios da Lógica Tensorial.

## Introdução

As redes de tensores lógicos (LTN) fornecem uma maneira poderosa de incorporar conhecimento simbólico em modelos de aprendizado profundo. Neste tutorial, você aprenderá a:

1. Representar regras como axiomas lógicos.
2. Traduzir axiomas em funções diferenciais utilizando "grounding".
3. Utilizar esses axiomas como funções de perda para orientar o aprendizado do modelo.

## Definição de Axiomas

Axiomas representam fatos ou regras que acreditamos serem verdadeiros dentro do nosso sistema. No LTN, esses axiomas podem ser representados por expressões como:

- `P(x)` → `Q(x)` (Se algo é P, então também é Q.)
- `∀x ∃y [R(x, y)]` (Para todo x, existe um y tal que R se aplica.)

Esses axiomas podem ser traduzidos em tensores e avaliados utilizando os conectivos lógicos fuzzy, como AND (min), OR (max), NOT (1-x), e operadores de quantificação (soma ou produto).

## Contexto do Problema: Classificação de Trens

### Atributos dos Trens

Os trens no problema possuem os seguintes atributos principais:

1. **Quantidade de Vagões:** Número entre 3 e 5.
2. **Quantidade de Cargas Diferentes:** Número entre 1 e 4.
3. **Atributos de Cada Vagão:**
    - Quantidade de eixos com rodas (2 ou 3).
    - Comprimento (curto ou longo).
    - Formato da carroceria: retângulo fechado, retângulo aberto, elipse, etc.
    - Quantidade de cargas no vagão (0 a 3).
    - Formato da carga (círculo, hexágono, retângulo ou triângulo).

Além disso, temos informações sobre a relação entre vagões:

- Proximidade de formatos específicos de carga (e.g., um vagão com carga circular próximo de outro com carga triangular).

O objetivo é usar esses atributos para classificar a direção do trem (leste ou oeste).

### Predicados e Axiomas

Baseando-se nos atributos, podemos definir os seguintes predicados:

1. `num_cars(t, nc)` - Representa a quantidade de vagões de um trem `t`.
2. `num_loads(t, nl)` - Representa a quantidade de tipos de carga no trem `t`.
3. `num_wheels(t, c, w)` - Representa a quantidade de rodas no vagão `c` do trem `t`.
4. `length(t, c, l)` - Representa o comprimento do vagão `c` (curto ou longo).
5. `shape(t, c, s)` - Representa o formato do vagão `c`.
6. `load_shape(t, c, ls)` - Representa o formato da carga no vagão `c`.
7. `next_crc(t, c, x)` - Indica se o vagão `c` possui um vizinho com carga circular.

Exemplo de axioma lógico:

- `∀t ∀c [short(c) ∧ closed_top(c) → east(t)]` (Se um vagão é curto e fechado, então o trem vai para o leste.)

## Implementação

### Passo 1: Preparando os Dados

Certifique-se de que os dados estejam preparados e sejam compatíveis com os predicados que você deseja modelar. Por exemplo, se você deseja testar `P(x)` → `Q(x)`, você precisará de exemplos rotulados de `P` e `Q`.

### Passo 2: Definindo Predicados e Funções

Os predicados, funções e constantes podem ser definidos como funções diferenciais:

```python
import torch
import torch.nn as nn

class Predicate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

num_cars = Predicate(input_dim=1)
num_loads = Predicate(input_dim=1)
short = Predicate(input_dim=1)
closed_top = Predicate(input_dim=1)
east = Predicate(input_dim=1)
```

### Passo 3: Expressando os Axiomas

Podemos agora expressar axiomas lógicos:

- **Implicação:** `short(c) ∧ closed_top(c) → east(t)` pode ser implementado como:

```python
def implication(p, q):
    return torch.min(1 - p, q)

def and_op(p, q):
    return torch.min(p, q)

short_values = short(c_data)
closed_top_values = closed_top(c_data)
east_values = east(t_data)
axiom_values = implication(and_op(short_values, closed_top_values), east_values)
```

- **Quantificadores:** Utilize agregadores fuzzy para quantificadores:

```python
def forall(values):
    return torch.min(values)

def exists(values):
    return torch.max(values)

forall_axiom = forall(axiom_values)
```

### Passo 4: Função de Perda

A satisfação dos axiomas pode ser usada como perda para otimizar o modelo:

```python
loss = 1 - forall_axiom
optimizer = torch.optim.Adam(list(num_cars.parameters()) + list(num_loads.parameters()) + list(east.parameters()), lr=0.01)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Exemplos

### Exemplo 1: Classificação Binária com Axiomas

Considere o seguinte:

- Dados: `x` com duas classes, `short(c)` e `closed_top(c)`.
- Regra: `short(c) ∧ closed_top(c) → east(t)`

Implemente o treinamento e visualize a satisfação da regra.

```python
for epoch in range(100):
    short_values = short(c_data)
    closed_top_values = closed_top(c_data)
    east_values = east(t_data)
    axiom_values = implication(and_op(short_values, closed_top_values), east_values)
    loss = 1 - forall(axiom_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item()}")
```

### Exemplo 2: Quantificador Existencial

Para `∃x load_shape(c, circle)`, assegure que pelo menos um elemento satisfaz o predicado:

```python
exists_axiom = exists(load_shape(c_data, circle))
loss = 1 - exists_axiom
```

## Conclusão

Neste tutorial, você aprendeu como:

1. Definir axiomas lógicos baseados em um problema real.
2. Traduzir atributos em predicados e axiomas diferenciais.
3. Implementar conectivos lógicos fuzzy e quantificadores.
4. Treinar modelos que satisfaçam regras lógicas em um problema de classificação de trens.

Explore mais combinações de axiomas e aplique em cenários reais! Para mais exemplos, confira a [documentação do LTNtorch](https://tommasocarraro.github.io/LTNtorch/).