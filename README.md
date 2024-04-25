# Fraude_Detection

Classification_Technics :Fichier qui contient les fonction des classifieurs ainsi que la fonction de test aui prend comme attributs les donnees non balance, le modele souhaite et la methode de balancement afin qu'elle retourne le resultat de performance de ce modele en utilisant 10-fold cross validation 



EDA :Fichier qui contient une analyse exploratoire de nos donnes 



Over_Sampling_tec :Fichier qui contient les fonctions de techniques de l over_sampling

Under_Sampling_techniques :Fichier qui contient les fonctions de techniques de l under_sampling

grid_search :Fichier qui contient la fonction qui performe le grid_search sur une pandas dataframe and a specific classifier 



DT :Fichier qui contient un test sur le modele Decision Tree en en exploitant les fonctions deja definis 


# Utilisation

Pour utiliser ce projet, il suffit de cloner le repertoire et de lancer le fichier Test.ipynb qui contient un exemple d'utilisation des fonction de resampling, de classification et de grid search.il suffit d'importer les fonctions necessaires et de les utiliser comme suit :

Exemple d'utilisation de la fonction de resampling :

```python
from Over_Sampling_tec import *
import pandas as pd

data = pd.read_csv('creditcard.csv')

data_resampled_SMOOTE = SMOOTE_tec(data)


```
For exmple this function returns a pandas dataframe with the resampled data using the SMOOTE technique

Exemple d'utilisation de la fonction de classification :

```python
from Classification_Technics import *

data = pd.read_csv('creditcard.csv')

model = DT_model(data)
#For testing the model zith a specific classifier and a specific resampling technique

test = Test_model(data, model ,SMOOTE_tec)




```

