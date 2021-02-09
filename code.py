import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import metrics

import matplotlib.pyplot as plt

import random

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

"""**Next few lines are needed to integrate the kaggle dataset ([Google Play Store Apps](https://www.kaggle.com/lava18/google-play-store-apps)) into google colab**"""

!pip install --quiet kaggle

from google.colab import files 
files.upload() #upload the json file that contains api key from kaggle account

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/.  

!chmod 600 ~/.kaggle/kaggle.json #altering permissions

!kaggle datasets download -d lava18/google-play-store-apps #this is the api of the dataset obtained from kaggle

from zipfile import ZipFile
zip_file= ZipFile('google-play-store-apps.zip')  #this downloaded zip file contains three csv file
data=pd.read_csv(zip_file.open('googleplaystore.csv'))  #we choose the googleplaystore.csv and load it into a dataframe called 'data' using pandas

data.head() #prints first 5 entries of the dataframe

data.info() #result shows there are 10841 entries in the dataframe , it also lists the columns present in the dataset

plt.figure(figsize=(7, 5))   
sns.heatmap(data.isnull(), cmap='viridis')      
data.isnull().any() #shows that Rating,Type,Content Rating,Current Ver and Android Ver cloumns have misisng value

data.isnull().sum() # shows the number of missing value in each column respectively

data['Rating'] = data['Rating'].fillna(data['Rating'].median()) #we replace the missing values of column Rating by the median of all Rating values

#we remove all the entries that have missing column values in Current Ver,Content Rating, Android Ver and Type
#we remove these entries corresponding to these columns as they have very few missing values
data = data[pd.notnull(data['Current Ver'])]
data = data[pd.notnull(data['Content Rating'])]
data = data[pd.notnull(data['Android Ver'])]
data = data[pd.notnull(data['Type'])]

data.isnull().sum() #to confirm that there are no further misisng values present

data.info() #we are left with 10829 entries now, out of 10841 (remaining are removed due to the presence of missing values)

"""## ***Studying each colum attribute in detail and cleaning it***

*1*. Size
"""

#function to convert MB and KB in bytes
def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x)*1000000
        return(x)
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x)*1000
        return(x)
    else:
        return None

data["Size"] = data["Size"].map(change_size) #update the Size column with these new values

data.Size.fillna(method = 'ffill', inplace = True) #filling null values

data.hist(column='Size', color='y') 
plt.xlabel('Size')
plt.ylabel('Frequency')

"""2. Installs

"""

data.Installs.value_counts()

#as linear regression deals with float values, we will remove any additional symbols present
data.Installs=data.Installs.apply(lambda x: x.strip('+'))  #remove symbol '+'
data.Installs=data.Installs.apply(lambda x: x.replace(',','')) #remove symbol ','

data['Installs'] = data['Installs'].astype(float)

data.Installs.value_counts()

"""3.Reviews"""

data.Reviews.str.isnumeric().sum() #checking if all 10829 Reviews are numeric values

data['Reviews'] = data['Reviews'].astype(int) #converting 'object' type Reviews to type 'int'

"""4.Rating"""

#checking the range of the values of the Rating column
print("Range: ", data.Rating.min(),"-",data.Rating.max())

data.Rating.hist();
plt.xlabel('Rating')
plt.ylabel('Frequency')

"""5.Type"""

data.Type.value_counts() #Prints the number of Free and Paid app

#Function that converts the Type value to '0' for free app and '1' for paid app
def type_cat(types):
    if types == 'Free':
        return 0
    else:
        return 1

data['Type'] = data['Type'].map(type_cat) #updated type value

"""6.Price"""

data.Price=data.Price.apply(lambda x: x.strip('$'))  #removing the symbol '$'
data['Price'] = data['Price'].astype(float)

data.Price.unique()

"""7.Category"""

data.Category.value_counts().plot(kind='bar',color='r')

#giving discrete dummy values to discrete Categories and adding them in a new Column 'Category_new'
CategoryL = data.Category.unique()
CategoryDict = {}
for i in range(len(CategoryL)):
    CategoryDict[CategoryL[i]] = i
data['Category_new'] = data['Category'].map(CategoryDict).astype(int)

"""8.Content Rating"""

data.columns = data.columns.str.replace(' ', '_') #for ex: replacing column name 'Content Rating' with 'Content_Rating'
data.Content_Rating.value_counts().plot(kind='bar')
plt.yscale('log')

#giving discrete dummy values to discrete Content Rating and updating them in the column
RatingL = data['Content_Rating'].unique()
RatingDict = {}
for i in range(len(RatingL)):
    RatingDict[RatingL[i]] = i
data['Content_Rating'] = data['Content_Rating'].map(RatingDict).astype(int)

"""9.Genres"""

#giving discrete dummy values to discrete Genres and adding them in a new Column 'Genre_new'
GenresL = data.Genres.unique()
GenresDict = {}
for i in range(len(GenresL)):
    GenresDict[GenresL[i]] = i
data['Genres_new'] = data['Genres'].map(GenresDict).astype(int)

"""10.Remaining"""

#dropping the columns that are not relevant for our linear regression
data.drop(labels = ['Last_Updated','Current_Ver','Android_Ver','App'], axis = 1, inplace = True)

"""***Final Database***"""

data.head() #first 5 entries of the updated dataframe

data.info() #checking if all the fields except 'Category' and 'Genres' are of either 'int64' or 'float64' type for regression

"""## **Linear Regression Model Building**

"""

print('Intercept: \n', model.intercept_) #value of b0X = data.drop(labels = ['Category','Rating','Genres'],axis = 1) #We remove the irrelevant columns
Y = data.Rating #Rating column is to be predicted and is assigned to Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30) #split 10829 entries into training sample(70%) amd test sample(30%)
model = LinearRegression() #model type will be linear regression

#fitting the model with the training set 
model.fit(X_train,Y_train)

print('Intercept: \n', model.intercept_) #value of b0

coeff_df = pd.DataFrame(model.coef_, ['Reviews','Size','Installs','Type', 'Price', 'Content_Rating','Category_new','Genres_new'], columns=['Coefficient'])  
coeff_df
#these coefficients are the values of b1,b2,b3....b8 respectively and they tell how the nature of dependence of Rating on these column attributes
#if the coefficient is positive/negative then Rating increases/decreases as the value of the attribute increases

Y_pred = model.predict(X_test) # Ratings are predicted using the regression model and saved in Y_pred
Y_pred

df1 = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred}) 
df2=df1.head(20)

df2.plot(kind='bar',figsize=(10,8)) #actual vs predicted Rating values
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='white')
plt.xlabel('app instance',color='black',fontsize=14)
plt.ylabel('rating',color='black',fontsize = 14)

plt.show()

plt.scatter(Y_test,Y_pred)  

plt.ylim(3.8,5) 
plt.xlim(0,5)

x = np.linspace(0, 5, 30)
plt.plot(x, x + 0,'-r', linestyle='solid')

plt.xlabel('Y Test (Actual Y)')
plt.ylabel('Predicted Y')

"""**Evaluation Metrics**"""

print ('Mean Squared Error: '+ str(metrics.mean_squared_error(Y_test,Y_pred)))
print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(Y_test,Y_pred)))
print ('Mean squared Log Error: '+ str(metrics.mean_squared_log_error(Y_test,Y_pred)))

"""# **Regression using statsmodels , Least Square method**"""

X = X.astype('float64') #explicitly converting all int type values to float type

import statsmodels.api as sm
X_opt = sm.add_constant(X) #constant column is needed in this method for b0 calculation
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #fitting the model 'regressor_OLS'
regressor_OLS.summary()

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_opt, Y, test_size = 0.2, random_state = 0) #repeating the train-test split this time in 80%-20% ratio
model2 = LinearRegression() #building new model
model2.fit(X_train2, Y_train2)
Y_pred2 = model2.predict(X_test2)
Y_pred2

"""**Some important graphs for the Regression model**"""

residuals= Y_test2-Y_pred2

# normalized residuals
model_norm_residuals = regressor_OLS.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
# absolute residuals
model_abs_resid = np.abs(residuals)
# leverage, from statsmodels internals
model_leverage = regressor_OLS.get_influence().hat_matrix_diag

plot1 = plt.figure()
plot1.axes[0] = sns.residplot(Y_pred2, residuals,
                          lowess=True,
                          scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot1.axes[0].set_title('Residuals vs Fitted')
plot1.axes[0].set_xlabel('Fitted values')
plot1.axes[0].set_ylabel('Residuals');

plot2 = plt.figure();
plt.scatter(model_leverage, model_norm_residuals, alpha=0.5);
sns.regplot(model_leverage, model_norm_residuals,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
plot2.axes[0].set_xlim(0, max(model_leverage)+0.01)
plot2.axes[0].set_ylim(-3, 5)
plot2.axes[0].set_title('Residuals vs Leverage')
plot2.axes[0].set_xlabel('Leverage')
plot2.axes[0].set_ylabel('Standardized Residuals');

