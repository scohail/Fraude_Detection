# Fraude_Detection

Classification_Technics :Fichier qui contient les fonction des classifieurs ainsi que la fonction de test aui prend comme attributs les donnees non balance, le modele souhaite et la methode de balancement afin qu'elle retourne le resultat de performance de ce modele en utilisant 10-fold cross validation 



EDA :Fichier qui contient une analyse exploratoire de nos donnes 



Over_Sampling_tec :Fichier qui contient les fonctions de techniques de l over_sampling



grid_search :Fichier qui contient la fonction qui performe le grid_search sur une pandas dataframe and a specific classifier 



DT :Fichier qui contient un test sur le modele Decision Tree en en exploitant les fonctions deja definis 


# Utilisation

Pour utiliser ce projet, il suffit de cloner le repertoire et de lancer le fichier main.py qui contient un exemple d'utilisation de la fonction de test qui prend comme attributs les donnees non balance, le modele souhaite et la methode de balancement afin qu'elle retourne le resultat de performance de ce modele en utilisant 10-fold cross validation

```python

import pandas as pd
from Classification_Technics import test
from Over_Sampling_tec import over_sampling
from grid_search import grid_search

# Load the data
data = pd.read_csv('creditcard.csv')

# Split the data into X and y
X = data.drop('Class', axis=1)
y = data['Class']

# Test the model
test(X, y, 'DT', 'SMOTE')


# Perform the grid search
grid_search(X, y, 'DT')

```