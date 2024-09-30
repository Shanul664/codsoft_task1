#TITANIC SURVIVAL PREDICTION

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

df=pd.read_csv("D:\Data Science Internship\Titanic-Dataset.csv")
df.head(10)

df.isnull().sum()

#Removing Cabin column
df.drop(['Cabin'], axis=1,inplace=True)

df.dropna(inplace=True)
df.info()

#Now no null column

survived_counts = df['Survived'].value_counts()
survived_counts

#Survival Proportion plot
plt.figure(figsize=(8, 6))
plt.pie(survived_counts, labels=['No', 'Yes'], autopct='%1.1f%%', colors=['#FF9999', '#66B2FF'], startangle=90)
plt.title('Survival Proportion')
plt.show()

#Survival Count Plot
plt.figure(figsize=(8, 6))
sns.barplot(x=survived_counts.index, y=survived_counts.values, palette='viridis')
plt.title('Survival Count')
plt.xlabel('Survived (1 = Yes, 0 = No)')
plt.ylabel('Number of Passengers')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

df['Pclass'].value_counts()

#Ticket class plot
sns.barplot(x=[1,2,3],y=df['Pclass'].value_counts(),palette='viridis')
plt.title("Ticket Class Count")
plt.ylabel('Number of tickets')
plt.xlabel('Ticket Class')
plt.show()

df.Sex.value_counts()

#Gender Plot of Male and Female
sns.barplot(x=['Male','Female'],y=df.Sex.value_counts(),palette='viridis')
plt.title("Gender Table")
plt.xlabel('Gender Type')
plt.ylabel('Gender Count')
plt.show()

df.Age.describe()

#Age Distribution Plot
sns.histplot(x=df['Age'],bins=20)
plt.title("Age Distribution Table")
plt.xlabel("Age")
plt.ylabel('Age Count')
plt.show()

df.SibSp.value_counts()

#Siblings Count Plot
sns.barplot(x=['0','1','2','3','4','5'],y=df.SibSp.value_counts(),palette='viridis')

df.Parch.value_counts()

#Parents/Childern Plot
sns.barplot(x=['0','1','2','3','4','5','6'],y=df.Parch.value_counts(),palette='viridis')

df.Fare.describe()

#Fare Distribution Plot
sns.histplot(df['Fare'], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Fair Price")

#Dectecting Outliers
Q1= df['Fare'].quantile(0.25)
Q3=df['Fare'].quantile(0.75)

IQR=Q3-Q1

upper=Q3 + 1.5*IQR
Lower=Q1 - 1.5*IQR

print(upper)
print(Lower)

df[df["Fare"]>70]

Embarked_counts=df['Embarked'].value_counts()
Embarked_counts

#Embarked Counts Plot
sns.barplot(x=['S','C','Q'],y=Embarked_counts,palette='viridis')

df.groupby(['Sex'])['Survived'].value_counts()

df.groupby(['Pclass'])['Survived'].value_counts()

df.groupby(['Pclass','Sex'])['Survived'].value_counts()

bins = [0, 18, 35, 50, 65, float('inf')]
labels = ['0-18', '19-35', '36-50', '51-65', '66+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

df[df['Survived']==0].groupby(['AgeGroup'])['Survived'].value_counts()

df.groupby(['AgeGroup','Sex'])['Survived'].value_counts()

df.groupby(['SibSp'])['Survived'].value_counts()

df.groupby(['Parch'])['Survived'].value_counts()

#Regression
df.info()
df.corr(numeric_only=True)
df.drop(['PassengerId','Ticket','Name','AgeGroup'],axis=1,inplace=True)

df.head()

encoder= LabelEncoder()
df['Sex']=encoder.fit_transform(df['Sex'])
df['Embarked']=encoder.fit_transform(df['Embarked'])
df['Survived']=encoder.fit_transform(df['Survived'])

X=df.drop(columns=["Survived"])
y=df.Survived

X
y

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=7)

model=LogisticRegression(max_iter=12000)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

model.score(X_train,y_train)*100
model.score(X_test,y_test)*100

rate=accuracy_score(y_test,y_pred)*100
print(f'{int(rate)}%')

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

modelsv=SVC()
modelsv.fit(X_train,y_train)

y_pred=modelsv.predict(X_test)

modelsv.score(X_train,y_train)*100

modelsv.score(X_test,y_test)*100

rate=accuracy_score(y_test,y_pred)*100
print(f'{int(rate)}%')

#KNN Classifer
ModelK=KNeighborsClassifier(n_neighbors=12)
ModelK.fit(X_train,y_train)

y_pred=ModelK.predict(X_test)

ModelK.score(X_train,y_train)*100

ModelK.score(X_test,y_test)*100

rate=accuracy_score(y_test,y_pred)*100
print(f'{int(rate)}%')

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

#Decision Tree Classifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

print(classifier.score(X_train,y_train)*100)

print(classifier.score(X_test,y_test)*100)

print(accuracy_score(y_test,y_pred)*100)