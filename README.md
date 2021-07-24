# Classifaction-algorithms-ML  
This is a simple implementation of all the regression models used in Machine Learning using python.

___   

# Installations 

```python
pip install numpy   
pip intall matplotlib   
pip install pandas   

```

>Note* The dataset used is named "Social_Network_Ads.csv"


## TABLE OF CONTENTS   
1. [Logistic Regression](##Logistic-Regression)
2. [KNN classification](##KNN-classification)

___

## Logistic Regression 

Logistic regression is a supervised learning classification algorithm used to predict the probability of a target variable. The nature of target or dependent variable is dichotomous, which means there would be only two possible classes.
In simple words, the dependent variable is binary in nature having data coded as either 1 (stands for success/yes) or 0 (stands for failure/no).   

### **Steps involved**   
1. Importing the libraries.
2. Importing the dataset.
3. Splitting the dataset to training and testing sets.
4. Feature Scaling.
5. Training the logistic regression model on training set.
6. Predicting a new result.
7. Predicting the test result.
8. Making the confusion matrix.
9. Visualizing the training set results.
10. Visualizing the testing set results.   
 
 >*Note: The dataset used in logistic regression classification model is 'Social_Network_Ads.csv'.   

 ### **Observation**   

 The confusion matrix observed,   
 |   63  |   5	|
|---	|---	|
|   8	|  24	|

Where,   
True positive = 63   
False negative = 5   
True negative = 24   
False positve = 8   

The overall accuracy of this model was observed to be 87 %.

### **Visualization of Output**  

![Logistic-Regression](assets/Linear%20Regression.jpg)  
 ___   

## KNN Classification   

K-Nearest Neighbors (KNN) is one of the simplest algorithms used in Machine Learning for regression and classification problem. KNN algorithms use data and classify new data points based on similarity measures (e.g. distance function).
Classification is done by a majority vote to its neighbors. The data is assigned to the class which has the nearest neighbors. As you increase the number of nearest neighbors, the value of k, accuracy might increase.   

### Algorithm   
Step 1: Choose the number of k-neighbors. Default/common = 5.   
Step 2: Take the k-nearest neighbor of the data point, according to Euclidean distance.   
```
Euclidean distance = sqrt[(x_2 - x_1)^2 + (y_2 - y_1)^2]
```   
Step 3: Among the k neighbors, count the number of data points in each category.    
Step 4: Assign the new data point to the category where you counted the most neighbors.   

### **Steps Involved**   
1. Importing the libraries.
2. Importing the dataset.
3. Splitting the dataset to training and testing sets.
4. Feature Scaling.
5. Training the Knn classidication model on training set.
6. Predicting a new result.
7. Predicting the test result.
8. Making the confusion matrix.
9. Visualizing the training set results.
10. Visualizing the testing set results.    

 ### **Observation**   

 The confusion matrix observed,   
 |   64  |   4	|
|---	|---	|
|   3	|  29	|

Where,   
True positive = 64  
False negative = 4   
True negative = 29   
False positve = 3   

The overall accuracy of this model was observed to be 93 %.   

### **Visualization of Output**  

![KNN](assets/KNN.jpg)
 ___   


