import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

historical_ranking = pd.read_excel('data/raw_data.xlsx', sheet_name='final')
all_year = list(set(historical_ranking['Year']))
historical_ranking.set_index(['Year','Name'], inplace=True)
print(historical_ranking.columns)
y = historical_ranking['Champion'] #feature column
X = historical_ranking.loc[:,historical_ranking.columns!='Champion'] #data set

#X_train,X_test, y_train, y_test = train_test_split(X, y, random_state=0)
year_to_test = 2017
all_year.remove(year_to_test)
X_train = X.loc[all_year]
X_test = X.loc[year_to_test]
y_train = y.loc[all_year]
y_test = y.loc[year_to_test]
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, y_train) #train the classifier
print(y_test)
print(knn.predict(X_test))
print(knn.score(X_test, y_test))

singer_2018 = pd.read_excel('data/raw_data.xlsx', sheet_name='Prediction')
singer_2018.set_index(['Year','Name'], inplace=True)
singer_2018['Predict_Champion'] = knn.predict(singer_2018)
print(singer_2018['Predict_Champion'])

