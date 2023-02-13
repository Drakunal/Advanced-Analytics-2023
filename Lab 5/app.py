import pandas as pd
import pickle
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder # Ordinal encoder
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

data = pd.read_csv('data2.csv')
print(data.head())
ord_enc = OrdinalEncoder()
data["outlook_code"] = ord_enc.fit_transform(data[["Outlook"]])
data['temperature_code'] = ord_enc.fit_transform(data[['Temperature']])
data['humidity_code'] = ord_enc.fit_transform(data[['Humidity']])
#print(data[["Outlook", "outlook_code"]].head(11))

X=data[['outlook_code', 'temperature_code', 'humidity_code', 'Windy']]  # Features
y=data['Play']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

#Create a Classifier
clf=RandomForestClassifier(n_estimators=1000)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


feature_imp = pd.Series(clf.feature_importances_).sort_values(ascending=False)
print(feature_imp)

# save the model to disk
filename = 'model.sav'
pickle.dump(clf, open(filename, 'wb'))