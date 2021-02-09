# Mobile-App-Success-and-Rating-Prediction-using-Machine-Learning
Mobile App Success and  Rating Prediction using Machine Learning

### Supervised this project carried by talented team of students Anushka Patil and Sreya Venkatesh

Mobile application development is a highly innovative software industry that has turned
into an extremely profitable business, with revenues only continuing to rise yearly. Due
to immense competition from around the world it is necessary for the app developers
to predict the success of their app and whether they are proceeding in the right
direction. As most of the apps in the play store are free the revenue generated by the
subscriptions, in-app purchases, in-app adverts are practically unknown. The success
of an app is usually determined by the user ratings and the number of installs rather
than the revenue it generates.

The app features of the apps available google play store are used which
are made available on Kaggle. Kaggle provides a rich source of information
regarding the apps which includes many attributes such as name, category, review,
rating, price etc. Based on the features offered and user reviews of the apps the
rating shall be predicted with the help of Python libraries and packages like sklearn,
statsmodel, seaborn, matplotlib etc.
## METHODOLOGY

![alt text](https://github.com/siddhaling/Mobile-App-Success-and-Rating-Prediction-using-Machine-Learning/blob/main/images/1.jpg)

Multiple linear regression attempts to model the relationship between two or more
explanatory(independent) variables and a response (dependent) variable by fitting a
linear equation to observed data.

## DATASET DESCRIPTION
The dataset, ‘Google Play Store Apps’ was obtained from Kaggle and used for this
study. No of observation (rows): 10840
Indepdendent variables
i. App: This contains the application name
ii. Category: Category of the app
iii. Reviews: No. of user reviews
iv. Size: Size of the app
v. Installs: Number of user installs
vi. Type: Paid or Free
vii. Price: Price of the app
viii. Content Rating: Age group the app is targeted at - Children / Mature 21+ /
Adult
ix. Genres: multiple genres (For eg, a game can belong to Music, Game, Family
genres.
x. Last Updated: Date when the app was last updated
xi. Current Ver: Current version of the app available on Play Store
xii. Android Ver: Min required Android version

Dependent variable: user rating of the app

‘non-null’ implies there are no missing values. Now, linear regression can be done on
numerical or continuous attributes but not on object type attributes such as app,
category, reviews, size, installs etc. Therefore, we would be converting these object
type variables into int values (0/1) in preprocessing section.

The following screenshot shows that rating, type, price, content rating, current ver
and android ver have missing values. We replace the missing Rating values by the median of all Rating values.

![alt text](https://github.com/siddhaling/Mobile-App-Success-and-Rating-Prediction-using-Machine-Learning/blob/main/images/2.jpg)

The size of all the apps in bytes. 

![alt text](https://github.com/siddhaling/Mobile-App-Success-and-Rating-Prediction-using-Machine-Learning/blob/main/images/3.jpg)

Histogram of ratings:

![alt text](https://github.com/siddhaling/Mobile-App-Success-and-Rating-Prediction-using-Machine-Learning/blob/main/images/4.jpg)

Frequency for the category of apps (showing the number of app per category).

![alt text](https://github.com/siddhaling/Mobile-App-Success-and-Rating-Prediction-using-Machine-Learning/blob/main/images/5.jpg)

Content rating:

![alt text](https://github.com/siddhaling/Mobile-App-Success-and-Rating-Prediction-using-Machine-Learning/blob/main/images/6.jpg)

The dataset is split into train and test set. The model is trained using train test and then values
are predicted using this model with test set as input. Train set: 70% and Test set: 30%.

we train the model using LinearRegression() function.

Residuals versus fits plot.

![alt text](https://github.com/siddhaling/Mobile-App-Success-and-Rating-Prediction-using-Machine-Learning/blob/main/images/7.jpg)

Model Fitting and coefficients

![alt text](https://github.com/siddhaling/Mobile-App-Success-and-Rating-Prediction-using-Machine-Learning/blob/main/images/8.jpg)

Regression equation

![alt text](https://github.com/siddhaling/Mobile-App-Success-and-Rating-Prediction-using-Machine-Learning/blob/main/images/9.jpg)

Predicted Vs Actual Values

![alt text](https://github.com/siddhaling/Mobile-App-Success-and-Rating-Prediction-using-Machine-Learning/blob/main/images/10.jpg)

Bar Graph Predicted Vs Actual Values

![alt text](https://github.com/siddhaling/Mobile-App-Success-and-Rating-Prediction-using-Machine-Learning/blob/main/images/11.jpg)

# Further Projects and Contact
www.researchreader.com

https://medium.com/@dr.siddhaling

Dr. Siddhaling Urolagin,\
PhD, Post-Doc, Machine Learning and Data Science Expert,\
Passionate Researcher, Deep Learning, Machine Learning and applications,\
dr.siddhaling@gmail.com
