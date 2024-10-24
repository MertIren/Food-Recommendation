# %% Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Importing dataset
dataset = pd.read_csv(r'C:\Users\merti\Desktop\Other\Programming\Python\W8Less\Market_Basket_Optimisation.csv', header = None)

tr = []
for record in dataset.values:
    tr.append([str(item) for item in record])

# %% Training the Apriori model
from apyori import apriori
rules = apriori(tr, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# %% Visualising the results
results = list(rules)
# %% Putting results into a Pandas DataFrame
def inspect(results):
    lhs =        [tuple(result[2][0][0])[0] for result in results]
    rhs =        [tuple(result[2][0][1])[0] for result in results]

    support =    [result[1] for result in results]
    confidence = [result[2][0][2] for result in results]
    lift =       [result[2][0][3] for result in results]

    return list(zip(lhs, rhs, support, confidence, lift))

dataFrame = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# %% Sorting dataframe
dataFrame.nlargest(n=10, columns='Lift')
# %%