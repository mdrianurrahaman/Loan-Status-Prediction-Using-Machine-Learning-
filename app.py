import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn import svm 
from sklearn.metrics import accuracy_score

### Data Collection and Preprocessing 
loan_dataset = pd.read_csv(r'C:\Users\ramiu\Desktop\Machine Learning Projects\Loan Status Prediction\train_u6lujuX_CVtuZ9i (1).csv')

print(type(loan_dataset))

print(loan_dataset.head())


###### statistical measure ###### 

print(loan_dataset.describe())

###### number of missing values in each column 

print(loan_dataset.isnull().sum())

### dropping all the missing values 

loan_dataset = loan_dataset.dropna()
print(loan_dataset.isnull().sum())

### label encoding 
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

######### printing the first 5 rows of the dataframe###### 

print(loan_dataset.head())

###### dependent coloumn values 

loan_dataset['Dependents'].value_counts()
print(loan_dataset)

##### replace the value 3+ to 4

loan_dataset = loan_dataset.replace(to_replace='3+',value=4)

######## dependent values 

print(loan_dataset['Dependents'].value_counts())

#### Data Visualization #### 

sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)
plt.show()

###### maritial status & loan status ######## 

sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)
plt.show()

# convert categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

print(loan_dataset.head())

###### separating the data and label ###### 

x = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1) 
y = loan_dataset['Loan_Status']

print(x)
print(y)

#######Train test split #### 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=2)
print(x.shape,x_train.shape,x_test.shape)

######## Training the model #######
####### support vector machine model #### 
classifier = svm.SVC(kernel='linear')


### traing the support vector 

classifier.fit(x_train,y_train)

#### model evaluation ### 


##### accuracy score on training data ###
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)
print('Accuracy on training data : ', training_data_accuracy)


# accuracy score on test data
x_test_prediction = classifier.predict(x_test)
test_data_accuray = accuracy_score(x_test_prediction,y_test)

print('Accuracy on test data : ', test_data_accuray)


