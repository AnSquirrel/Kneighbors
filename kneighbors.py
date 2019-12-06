
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

acoh = [[12, 500], [15, 520], [20, 400], 
        [22, 450], [25, 520], [30, 550], 
        [38, 500], [45, 450], [48, 360], 
        [52, 330], [55, 250], [58, 220]]
alcohol = np.array(acoh)

liqu = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
liqueur = np.array(liqu)

liqu_name = ['red', 'white']

a_train, a_test, l_train, l_test = train_test_split(
    alcohol, liqueur, random_state=0)

print('Train And Test Alcohol:\n', a_train, 2 * '\n', a_test, '\n')
print('Train And Test Liqueur:\n', l_train, 2 * '\n', l_test, '\n')

kneighbors = KNeighborsClassifier(n_neighbors=1)
print('KNeighbors Model:\n', kneighbors)

kneighbors.fit(a_train, l_train)

l_predict = kneighbors.predict(a_test[2])
print('Predict Liqueur:', l_predict)
print('Liqueur Name:', liqu_name[l_predict])
