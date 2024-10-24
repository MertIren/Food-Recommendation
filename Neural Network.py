# %%
import numpy as np
import pandas as pd

# %%
dataset = pd.read_csv(r'C:\Users\merti\Desktop\Other\Programming\Python\W8Less\Market_Basket_Optimisation.csv', header = None)

tr = []
for record in dataset.values:
    tr.append([str(item) for item in record if str(item) != 'nan'])


# %%
uniques = []
for transaction in tr:
    for item in transaction:
        if item not in uniques:
            uniques.append(item)
# %%
key = {uniques[i]: i for i in range(len(uniques))}

def convert(transaction, key):
    dim = len(key)
    vec = np.zeros(dim)

    if transaction == [[]]:
        transaction = []

    for item in transaction:
        vec[key[item]] = 1

    return vec
# %%
def split(transaction):
    splits = []

    if len(transaction) == 1:
        splits.append([transaction[0:], []])
        return splits
    for i in range(len(transaction)):
        splits.append([transaction[0:i] + transaction[i+1:], transaction[i]])

    return splits
# %%
X_y  = []

for transaction in tr:
    for spl in split(transaction):
        X_y.append(spl)

X = [item[0] for item in X_y]
y = [[item[1]] for item in X_y]
# %%
X_vec = np.array([convert(transaction, key) for transaction in X])
y_vec = np.array([convert(transaction, key) for transaction in y])
# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(300, input_shape=(120,), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(120, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

# %%
model.fit(X_vec, y_vec, epochs = 50, batch_size = 32)
# %%
model.save_weights('best_NN_unbiased.hdf5')
# %%
model.load_weights('best_NN.hdf5')
# %%
model.evaluate(X_vec, y_vec)
# %%
predictions = ['honey']
predictions = np.array([convert(predictions, key)])

result = model.predict([predictions])
order = np.argsort(-result)
for i in range(10):
    print(uniques[order[0][i]])
# %%
