import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
dataset=pd.read_csv('liver_patient_analysis.csv')
dataset['Gender'].fillna(0,inplace=True)
dataset['Albumin_and_Globulin_Ratio'].fillna(dataset['Albumin_and_Globulin_Ratio'].mean(),inplace=True)
X=dataset.iloc[:,:4]
def convert_to_int(word):
        word_dict={'Female':1,'Male':2}
        return word_dict[word]
X['Gender']=X['Gender'].apply(lambda x : convert_to_int(x))
y=dataset.iloc[:,-1]
from sklearn.linear_model import LogisticRegression
model = LogisticRegression();
model.fit(X,y)
pickle.dump(model,open('model2.pkl','wb'))
model=pickle.load(open('model2.pkl','rb'))
print(model.predict([[0,2,6,8]]))
