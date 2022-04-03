# Prediction-of-Passengers-Survival-in-Titanic
Titanic was a British liner ship which sank after colliding with an iceberg. It contained around 2225 passengers, out of which more than 1500 died in the collision. The project is based on predicting the people died / survived that accident using Machine Learning Algorithms and python. 

To create a machine learning model on the Titanic dataset, which is used by many people all over the world. It provides information on the fate of passengers on the Titanic – whether they survived or not, and to visualise according to economic status (class), sex, age and survival.


# Objective:
To create a machine learning model on the Titanic dataset, which is used by many people all over the world. It provides information on the fate of passengers on the Titanic – whether they survived or not, and to visualise according to economic status (class), sex, age and survival.


# Procedure:
## 1. Importing Dataset:
## 2. Data Vizualization:
## 3. Data Cleaning:
## 4. Converting the unique values of some column into Numerics:
## 5. Splitting the data into independent ‘X’ and dependent ‘Y’ data sets.
## 6. Split the dataset into 80% Training set and 20% Testing set.
## 7. Feature Scaling
## 8. Model training (6 models)
a) Logistic Regression,
b) K Neighbors Classifier,
c) Support Vector Machine
d) Naïve Bayes,
e) Decision Tree Classifier
f) Random Forest Classifier
## 9. Calculating Training accuracy and Testing accuracy.
## 10. Finding a Conclusion.


# Some Basic Understanding of Machine Learning Models used in the project

## 1) Logistic Regression:
Logistic regression is a type of Supervised Learning. It predicts the output of a categorical dependent variable. Therefore, the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.

## 2) K Neighbors Classifier:
K Neighbors Classifier is a type of Supervised Learning. It assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.

## 3) Support Vector Machine:
It is also a type of Supervised Learning. The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.

## 4) Gaussian Naïve Bayes:
It helps in building the fast machine learning models that can make quick predictions. It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.

## 5) Decision Tree Classifier:
It is another type of Supervised Learning. A Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.

## 6) Random Forest Classifier:
It is a type of Supervised Learning. Instead of relying on one decision tree, the random forest classifier takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output. The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.


# Confusion Matrix in Machine Learning 
The confusion matrix is a matrix used to determine the performance of the classification models for a given set of test data. It can only be determined if the true values for test data are known.

### True Negative: 
Model has given prediction No, and the real or actual value was also No.
### True Positive: 
The model has predicted yes, and the actual value was also true.
### False Negative: 
The model has predicted no, but the actual value was Yes, it is also called as Type-II error.
### False Positive: 
The model has predicted Yes, but the actual value was No. It is also called a Type-I error.


# Conclusion of the Project
Random Forest Classifier is selected as the best model to predict survivors with 97 % training accuracy and 80.4% testing accuracy. We will ignore Decision Tree accuracy because no model gives so perfect results and accuracy in any prediction.


