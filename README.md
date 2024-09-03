## Apresentação

Este repositório apresenta uma análise detalhada dos dados contidos no arquivo `startup data.csv`, conforme as instruções da ementa do processo seletivo para o Laboratório de Pesquisa de Software do CESUPA.

A análise será dividida em algumas etapas, incluindo a identificação dos dados, o tratamento de dados nulos, a remoção de fileiras inválidas e duplicadas, a separação de dados categóricos e numéricos, e a elaboração de um modelo preditivo para determinar se uma startup será bem-sucedida ou não, com base nos dados inseridos pelo usuário.

---
## Introdução

Para estudo da tabela, se foi utilizado algumas bibliotecas populares de Python, como Seaborn e Matplotlib para projeção de gráficos e Pandas para tratamento dos dados.

```python
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
```
### Identificação dos dados

Para começar nossa análise, é essencial examinarmos as colunas presentes no conjunto de dados e compreendermos o significado de cada uma delas e as relações presentes entre si.

```python
df = pd.a('startup data.csv')
df.info()
```

```
<class 'pandas.core.frame.DataFrame'> RangeIndex: 923 entries, 0 to 922 Data columns (total 49 columns):

# Column Non-Null Count Dtype
--- ------ -------------- -----
0 Unnamed: 0                923 non-null int64
1 state_code                923 non-null object
2 latitude                  923 non-null float64
3 longitude                 923 non-null float64
4 zip_code                  923 non-null object
5 id                        923 non-null object
6 city                      923 non-null object
7 Unnamed: 6                430 non-null object
8 name                      923 non-null object
9 labels                    923 non-null int64
10 founded_at               923 non-null object
11 closed_at                335 non-null object
12 first_funding_at         923 non-null object
13 last_funding_at          923 non-null object
14 age_first_funding_year   923 non-null float64
15 age_last_funding_year    923 non-null float64
16 age_first_milestone_year 771 non-null float64
17 age_last_milestone_year  771 non-null float64
18 relationships            923 non-null int64
19 funding_rounds           923 non-null int64
20 funding_total_usd        923 non-null int64
21 milestones               923 non-null int64
22 state_code.1             922 non-null object
23 is_CA                    923 non-null int64
24 is_NY                    923 non-null int64
25 is_MA                    923 non-null int64
26 is_TX                    923 non-null int64
27 is_otherstate            923 non-null int64
28 category_code            923 non-null object
29 is_software              923 non-null int64
30 is_web                   923 non-null int64
31 is_mobile                923 non-null int64
32 is_enterprise            923 non-null int64
33 is_advertising           923 non-null int64
34 is_gamesvideo            923 non-null int64
35 is_ecommerce             923 non-null int64
36 is_biotech               923 non-null int64
37 is_consulting            923 non-null int64
38 is_othercategory         923 non-null int64
39 object_id                923 non-null object
40 has_VC                   923 non-null int64
41 has_angel                923 non-null int64
42 has_roundA               923 non-null int64
43 has_roundB               923 non-null int64
44 has_roundC               923 non-null int64
45 has_roundD               923 non-null int64
46 avg_participants         923 non-null float64
47 is_top500                923 non-null int64
48 status                   923 non-null object
```

<img src="IMG/Fig. 1.png" alt="Figura 1"/>

> Figura 1. Exemplo de cada coluna

As colunas `state_code`, `latitude`, `longitude`, `zip_code`, `city`, `Unnamed:6`, `state_code.1`,
`is_CA`, `is_NY`, `is_MA`, `is_TX` e `is_otherstate` contêm informações relacionadas à localização das startups documentadas. A coluna `Unnamed: 6` é uma concatenação do endereço completo, combinando `city`, `state_code` e `zip_code`.

As colunas `Unnamed: 0`, `id`, `name` e `object_id` são essenciais para a identificação única de cada startup, seja por nome ou por identificadores numéricos. Já `category_code` e as dez colunas subsequentes categorizam as startups de acordo com suas áreas de atuação.

As colunas `founded_at`, `closed_at`, `first_funding_at`, `last_funding_at`, `age_first_funding_year`, `age_last_funding_year`, `age_first_milestone_year` e `age_last_milestone_year` referem-se às datas de financiamento, bem como às datas de abertura e encerramento das empresas.

As colunas `labels`, `relationships`, `funding_rounds`, `funding_total_usd`, `milestones`, `has_VC`, `has_angel`, `has_round`, `avg_participants`, e `is_top500` representam valores numéricos e indicadores chave relacionados ao financiamento das startups. Elas fornecem informações sobre o número de rodadas de investimento, o financiamento total, os marcos atingidos, a presença de investidores como venture capital (VC) e anjos investidores, bem como o número médio de participantes. Por fim, esses dados culminam na coluna `status`, que indica se a startup foi adquirida ou fechada.

### Mapa de correlação

Agora que identificamos as funções de cada coluna na tabela, uma próxima etapa importante é separar os dados em categorias distintas: numéricos e categóricos. 
```python
df['status']=df['status'].map({'acquired': 1, 'closed': 0}).astype(int)
df_categorical = df.select_dtypes(include=['object'])
df_numerical = df.drop(df_categorical.columns, axis=1)
```
Essa separação nos permitirá analisar como os dados numéricos interagem entre si, possibilitando a identificação de padrões, correlações e influências entre as variáveis numéricas. 
```python
corr = df_numerical.corr()
plt.figure(figsize=(25,10))
sns.heatmap(corr, annot=True, cmap='rocket')
plt.show()
```
Com isso, podemos obter insights valiosos sobre como aspectos como financiamento, número de participantes e marcos alcançados impactam o desempenho e o status final das startups.

<img src="IMG/Fig. 2.png" alt="Figura 2"/>

  > Figura 2. Mapa de calor de correlação

No mapa de calor, é interessante observar que há quatro pontos de maior atividade localizados nas extremidades. Essa concentração pode ser atribuída à correlação entre as datas de financiamento e a presença de diferentes rodadas de investimento. 

Outro ponto importante é que há uma correspondência de 1 para 1 entre `label` e `status`. Para verificar se as duas tabelas realmente possuem os mesmos valores (e, consequentemente, não precisam repetir as informações), podemos analisar se existem colunas onde esses valores não sejam equivalentes.

```python
df[df['labels'] != df['status']].shape
```
Resultados:
`(0, 49)`

Assim podemos deduzir que labels funciona apenas como um indicador booleano da coluna status pré-tratamento. Logo podemos largar essa tabela:
```python
df.drop(['labels'], axis=1, inplace=True)
```

### Tratamento de dados nulos

```python
missing=pd.DataFrame(df.isnull().sum(),columns=["Null Values"])
missing["% Missing Values"]=(df.isna().sum()/len(df)*100)
missing = missing[missing["% Missing Values"] > 0]
missing
```

| index                       | Null Values | % Missing Values     |
| --------------------------- | ----------- | -------------------- |
| Unnamed: 6                  | 493         | 53\.412784398699884  |
| closed\_at                  | 588         | 63\.705308775731304  |
| age\_first\_milestone\_year | 152         | 16\.468039003250272  |
| age\_last\_milestone\_year  | 152         | 16\.468039003250272  |
| state\_code\.1              | 1           | 0\.10834236186348861 |

Na tabela gerada pela célula de código acima, podemos ver que `Unnamed: 6`, nossa coluna de endereços concatenados se encontra com mais da metade das fileiras faltando e `closed_at` faltando em 63%, possivelmente estando relacionada ao número de startups que foram adquiridas.

Além dessas duas colunas, temos `age_first_milestone_year` e `age_last_milestone_year`, que estão faltando aparentemente nas mesmas fileiras. Por fim temos `state_code.1`, possivelmente uma coluna secundaria que se referia a coluna de `state_code`, logo tornando a mesma redundante.

```python
df.drop('state_code.1',axis=1,inplace=True)
```

Para tratamento dos dados que estão faltando, podemos preencher as colunas `Unnamed: 6`,  `age_first_milestone_year` e `age_last_milestone_year`,  com o endereço concatenado no caso de `Unnamed: 6` e com 0 nos anos de marcos da startup.

```python
df['Unnamed: 6'].fillna((df['city'] + ' ' + df['state_code'] + ' ' + df['zip_code']), inplace=True)
df['Unnamed: 6'].head(10)
```

```python
df['age_first_milestone_year'].fillna(0, inplace=True)
df['age_last_milestone_year'].fillna(0, inplace=True)
```

Para a coluna `closed_at` podemos verificar se existem fileiras onde a data de fechamento seja nula e o `status` da startup seja 0. Caso não tenha, podemos apenas substituir as entradas nulas por uma data placeholder a fins de mostrar que a startup não fechou durante a elaboração da planilha (uma data após a ultima entrada válida de encerramento).

```python
df['first_funding_at']=pd.to_datetime(df['first_funding_at'])
df['last_funding_at']=pd.to_datetime(df['last_funding_at'])

df['closed_at']=pd.to_datetime(df['closed_at'])
df['founded_at']=pd.to_datetime(df['founded_at'])
df['closed_at'].max() # Resposta: Timestamp('2013-10-30 00:00:00')
```

```python
df[(df['closed_at'].isnull()) & (df['status'] == 0)].shape # Resultado: (0, 47)
df['closed_at'].fillna('2014-01-01', inplace=True)
```

### Tratameto de discrepâncias

Para ver melhor as discrepâncias numéricas em nosso dataset, usamos o método `describe()` nas colunas de valor numérico.

```python
describeNum = df.drop(['latitude','longitude'],axis=1)
# Dividindo funding total por milhões para facilitar análise
describeNum['funding_total_usd'] = describeNum['funding_total_usd']/1_000_000
describeNum = describeNum.describe(include =['float64', 'int64', 'float', 'int'])
describeNum.T.style.background_gradient(cmap='rocket',low=0.2,high=2.9)
```

<img src="IMG/Fig. 3.png" alt="Figura 3"/>
  
  > Figura 3. - Colunas relevantes em describeNum

Aqui podemos ver que as datas de fundação e de marcos tem fileiras com valores negativos, o que torna essas startups nulas para nossa análise, então podemos apenas largar essas fileiras.

```python
df[['age_first_funding_year', 'age_last_funding_year', 'age_first_milestone_year', 'age_last_milestone_year']].plot(figsize=(15, 5))
```
<img src="IMG/Fig. 4.1.png" alt="Figura 4.1"/>

  > Figura 4.1 Idades de fundação/marcos das startups. Contém entradas com idades negativas.

<img src="IMG/Fig. 4.2.png" alt="Figura 4.2"/>

  > Figura 4.2 Investimentos totais nas startups

```python
df.drop(df[(df['age_first_funding_year'] < 0 )   | (df['age_last_funding_year'] < 0) | (df['age_first_milestone_year'] < 0 ) | (df['age_last_milestone_year'] < 0 )].index,axis=0, inplace=True)
```

Além disso, temos como outra coluna atípica a coluna de financiamento total, com uma derivação padrão enorme comparados ao resto do dataset. Podemos remover esses 'outliers' impondo limites superiores e inferiores para `funding_total_usd`.

```python
Q1 = df['funding_total_usd'].quantile(0.25)
Q3 = df['funding_total_usd'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

funding_total_outliers = df[(df['funding_total_usd'] < lower_bound) | (df['funding_total_usd'] > upper_bound)]

print(funding_total_outliers.shape[0]) # 8 fileiras
df.drop(funding_total_outliers.index,inplace=True)
```

Repetindo a célula de `describeNum` temos:
<img src="IMG/Fig. 5.png" alt="Figura 5"/>
> Figura 5. Dataset numérico normalizado

Por fim vamos ver como esta o dataset categórico
```python
describeNum = df.describe(include=["O"])
describeNum.T
```

| index          | count | unique | top                    | freq |
| -------------- | ----- | ------ | ---------------------- | ---- |
| state\_code    | 923   | 35     | CA                     | 488  |
| zip\_code      | 923   | 382    | 94107                  | 30   |
| id             | 923   | 922    | c:28482                | 2    |
| city           | 923   | 221    | San Francisco          | 128  |
| Unnamed: 6     | 923   | 401    | San Francisco CA 94107 | 30   |
| name           | 923   | 922    | Redwood Systems        | 2    |
| category\_code | 923   | 35     | software               | 153  |
| object\_id     | 923   | 922    | c:28482                | 2    |

A unica atipicidade que encontramos por aqui é a presença de uma fileira duplicada, que pode ser facilmente removida.
```python
df[df['id'] == 'c:28482'].shape # (2, 47)
df.drop(df[df['id'] == 'c:28482'].index[1], axis=0,inplace=True)
```

Por fim, vamos re-declarar o data frame categórico e numérico com nossos dados tratados, renomeando algumas colunas para explicar melhor seus propósitos.
```python
df.rename(columns={'Unnamed: 0': 'index', 'Unnamed: 6': 'full_address', 'status': 'is_acquired'}, inplace=True)

df_categorical = df.select_dtypes(include=['object'])
df_numerical = df.drop(df_categorical.columns, axis=1)
```
### Análise Gráfica de startups bem sucedidas

Por fim, é bom destacar que dois fatores que influenciam muito o desempenho de uma startup é a forma que ela é financiada, seja por investidores de capital de risco ou anjo investidores, se foi por rounds de investimento ou se foi por investimento "seed".

```python
df['has_RoundABCD'] = np.where((df['has_roundA'] == 1) | (df['has_roundB'] == 1) |(df['has_roundC'] == 1) | (df['has_roundD'] == 1), 1, 0)

df['has_Investor'] = np.where((df['has_VC'] == 1) | (df['has_angel'] == 1), 1, 0)

df['has_Seed'] = np.where((df['has_RoundABCD'] == 0) & (df['has_Investor'] == 1), 1, 0)

df['invalid_startup'] = np.where((df['has_RoundABCD'] == 0) & (df['has_Investor'] == 0), 1, 0)
```

```python
seeded_success = df[(df['is_acquired'] == 1) & (df['has_Seed'] == 1)].shape[0]
seeded_failure = df[(df['is_acquired'] == 0) & (df['has_Seed'] == 1)].shape[0]
seeded_labels = ['Adquiridas (' + str(seeded_success) + ')', 'Fechada(' + str(seeded_failure) + ')']
sides_seeded = [seeded_success, seeded_failure]
colors = ['green', '#CB1111']

round_success = df[(df['is_acquired'] == 1) & (df['has_RoundABCD'] == 1)].shape[0]
round_failure = df[(df['is_acquired'] == 0) & (df['has_RoundABCD'] == 1)].shape[0]
round_labels = ['Adquiridas (' + str(round_success) + ')', 'Fechada(' + str(round_failure) + ')']
sizes_round = [round_success, round_failure]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].pie(sides_seeded, labels=seeded_labels, colors=colors, autopct='%5.1f%%', startangle=90, wedgeprops=dict(width=0.6))
axs[0].axis('equal')
axs[0].set_title('Taxa de sucesso de startups \'semeadas\'')

axs[1].pie(sizes_round, labels=round_labels, colors=colors, autopct='%5.1f%%', startangle=90, wedgeprops=dict(width=0.6))
axs[1].axis('equal')
axs[1].set_title('Taxa de sucesso de startups em rounds')

plt.show()
```
<img src="IMG/Fig. 6.1.png" alt="Figura 6.1"/>

  > Figura 6.1 Gráficos de Sucesso/Falha por tipo de startup

```python
category_counts['Acquired'] = df[df['is_acquired'] == 1].groupby('category_code')['is_acquired'].count()
category_counts['Closed'] = df[df['is_acquired'] == 0].groupby('category_code')['is_acquired'].count()

category_counts.fillna(0, inplace=True)

category_counts['Total'] = category_counts['Acquired'] + category_counts['Closed']

category_counts['Acquired_prop'] = category_counts['Acquired'] / category_counts['Total']
category_counts['Closed_prop'] = category_counts['Closed'] / category_counts['Total']

category_counts = category_counts.sort_values(by='Total', ascending=False)

fig, axs = plt.subplots(2, 1, figsize=(15, 10))

labels_top20 = [f'{label} ({int(count)})' for label, count in zip(category_counts.index[:20], category_counts['Total'][:20])]

axs[0].bar(labels_top20, category_counts['Acquired_prop'][:20], color='green', label='Acquired')
axs[0].bar(labels_top20, category_counts['Closed_prop'][:20], bottom=category_counts['Acquired_prop'][:20], color='#CB1111', label='Closed')


axs[0].set_title('Comparação Proporcional de Startups Bem-Sucedidas e Encerradas em Ordem Decrescente')
axs[0].set_xlabel('Categoria')
axs[0].set_ylabel('Proporção')
axs[0].tick_params(axis='x', rotation=45)
axs[0].legend()

labels_resto = [f'{label} ({int(count)})' for label, count in zip(category_counts.index[20:], category_counts['Total'][20:])]

axs[1].bar(labels_rest, category_counts['Acquired_prop'][20:], color='green', label='Acquired')
axs[1].bar(labels_rest, category_counts['Closed_prop'][20:], bottom=category_counts['Acquired_prop'][20:], color='#CB1111', label='Closed')
axs[1].set_title('Comparação Proporcional de Startups Bem-Sucedidas e Encerradas em Ordem Decrescente')
axs[1].set_xlabel('Categoria')
axs[1].set_ylabel('Proporção')
axs[1].tick_params(axis='x', rotation=45)
axs[1].legend()

plt.tight_layout()
plt.show()
```
<img src="IMG/Fig. 6.2.png" alt="Figura 6.2"/>

  > Figura 6.2 Gráficos de Sucesso/Falha por categoria (proporcionais)

```python
plt.figure(figsize=(12, 6))

# Histograma para aberturas de startup
plt.subplot(1, 2, 1)
plt.hist(df['founded_at'], bins=50)
plt.title('Histograma das Datas de Fundação das Startups')
plt.xlabel('Data de Fundação')
plt.ylabel('Número de Startups')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)

# Histograma para fechamentos de startup
plt.subplot(1, 2, 2)
plt.hist(df['closed_at'].drop(df[df['closed_at'] == '01/01/2014'].index), bins=50)
plt.title('Histograma das Datas de Encerramento das Startups')
plt.xlabel('Data de Encerramento')
plt.ylabel('Número de Startups')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```
<img src="IMG/Fig. 6.3.png" alt="Figura 6.3"/>
  
  > Figura 6.3 Histograma de datas

# Insights Finais  

1. Um fator que fortemente impacta o desempenho de uma startup é se ela é "semeada", ou seja financiada por incubadora ou se ela é financiada por rounds, contando com o investimento após metas. Uma hipotese do porquê disso acontecer é que startups incubadas contam apenas com uma ementa inicial para o investimento, já startups por rounds também contam com a mentoria dos investidores e uma visão mais dinâmica do mercado.

2. O mercado de startups, ao ver desse dataset, é fortemente influênciado pela área de TI, com a grande maioria das startups sendo baseadas no desenvolvimento de software e desenvolvimento web, com a maioria do top 10 sendo relacionadas a área computacional.

3. Grande parte das startups nesse dataset foram fundadas durante a virada do milênio até 2005, com o número de startups que abriram drasticamente diminuindo depois do início da recessão econômica de 2008, juntamente ao crescimento do número de startups que fecharam durante esse período (2008-2013).
