import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

class Singer_prediction(object):
    def __init__(self, data_path, historical_sheet,test_sheet):
        self.historical_ranking = pd.read_excel(data_path, sheet_name=historical_sheet)
        self.all_year = list(set(self.historical_ranking['Year']))
        self.historical_ranking.set_index(['Year','Name'], inplace=True)
        self.singer_2018 = pd.read_excel(data_path, sheet_name=test_sheet)
        self.singer_2018.set_index(['Year','Name'], inplace=True)
        self.X_train, self.X_test, self.y_train, self.y_test = self.create_data_set()

    def create_data_set(self, year_to_test = 2016):
        y = self.historical_ranking['Champion'] #feature column
        X = self.historical_ranking.loc[:,self.historical_ranking.columns!='Champion'] #data set
        self.all_year.remove(year_to_test)
        return X.loc[self.all_year],X.loc[year_to_test], y.loc[self.all_year], y.loc[year_to_test]
        #return train_test_split(X, y, random_state=0)

    def KNN_prediction(self,nn=2):#k-nearest neighbor
        knn = KNeighborsClassifier(n_neighbors = nn)
        knn.fit(self.X_train, self.y_train) #train the classifier
        return knn.predict(self.singer_2018)

    def random_forest_prediction(self,n_est=10):#random forest
        rf = RandomForestClassifier(n_estimators=n_est)
        rf.fit(self.X_train, self.y_train)
        return rf.predict(self.singer_2018)

    def logistic_regression_prediction(self):#logistic regression
        lgr = LogisticRegression()
        lgr.fit(self.X_train, self.y_train)
        return lgr.predict(self.singer_2018)

    def svm_prediction(self):#support vector machine
        supporvm = svm.SVC()
        supporvm.fit(self.X_train, self.y_train)
        return supporvm.predict(self.singer_2018)

if __name__ == "__main__":
    predict_champ = Singer_prediction('data/raw_data.xlsx','final','Prediction')
    singer_prediction_2018 = pd.DataFrame(index=predict_champ.singer_2018.index)
    #run prediction using different models
    singer_prediction_2018['Predict_Champion_KNN'] = predict_champ.KNN_prediction()
    singer_prediction_2018['Predict_Champion_random_forest'] = predict_champ.random_forest_prediction(2)
    singer_prediction_2018['Predict_Champion_logistic_regression'] \
        = predict_champ.logistic_regression_prediction()
    singer_prediction_2018['Predict_Champion_svm'] \
        = predict_champ.svm_prediction()
    print(singer_prediction_2018)
    singer_prediction_2018.to_csv('data/prediction_result.csv',index=True)
