# Need to install apyori first
!pip install apyori

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from apyori import apriori
from wordcloud import WordCloud

#Import dataset
dataset = pd.read_csv('/kaggle/input/market-basket-optimization/Market_Basket_Optimisation.csv', header = None)

dataset.shape

dataset.sample(5)

plt.figure(figsize=(15,15))
wc = WordCloud(background_color = 'black', width = 1500,  height = 1500, max_words = 100).generate(str(dataset[0]))
plt.imshow(wc)
plt.axis('off')
plt.title('Popular Items in Market Basket')
plt.show()

plt.figure(figsize=(20,10))
color = plt.cm.cool(np.linspace(0, 1, 40))
dataset[0].value_counts().head(40).plot.bar(color = color)
plt.title('Market Analysis:Most sold products', fontsize = 30)
plt.xticks(rotation = 90 )
plt.show()

# Data preprocessing
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Apriori Model

# Training the Apriori model on the dataset
rules_1 = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Make a list of the rules
results_1 = list(rules_1)

# Put the results well organised into a Pandas DataFrame
def inspect_apriori(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame_1 = pd.DataFrame(inspect_apriori(results_1), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])



## Displaying the results sorted by descending lifts
resultsinDataFrame_1.nlargest(n = 5, columns = 'Lift')

# Training the Eclat model on the dataset
rules_2 = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Make a list of the rules
results_2 = list(rules_2)

# Put the results well organised into a Pandas DataFrame
def inspect_eclat(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame_2 = pd.DataFrame(inspect_eclat(results_2), columns = ['Product 1', 'Product 2', 'Support'])

## Displaying the results sorted by descending supports
resultsinDataFrame_2.nlargest(n = 5, columns = 'Support')

