Perform a predictive modeling analysis using the decision tree, k-NN techniques, logistic regression and SVM (explore how well model performs for several different hyper-parameter values). A brief overview of your predictive modeling process, explorations, and discussion of results. I you present information about the model “goodness” (confusion matrix, predictive accuracy, precision, recall, f-measure) and briefly discuss ROC and lift curves.


```python
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import metrics,neighbors, preprocessing,tree, linear_model, naive_bayes, svm
from sklearn.model_selection import cross_validate, train_test_split, validation_curve,GridSearchCV, KFold, StratifiedKFold,cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
import scikitplot
```

### Import the dataset into Pandas Dataframe


```python
dataFile = pd.read_csv('wdbc.data', header=None)
dataFile.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>

### Converting the target variable into Binary Form. 'M' = 1 and 'B' = 0 and normalizing the data.


```python
le = preprocessing.LabelEncoder()
le.fit(dataFile[1])
dataFile[1] = le.transform(dataFile[1]) 

X = dataFile.iloc[:,2:]
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, dataFile[1], test_size=0.3, random_state=4)
```

### Creating dictionaries 'Performance' and Solver-Penalty combination of the Logistical Regression Hyperparameters and a list to hold regularization constants


```python
# This will hold the performance measure and model output paramters ( precision, recall, accuract etc.)
performanceFalseNegatives = {}
performanceF1Score = {}
modelParams = {}
#Creating a dictionary with optimization algorithms and supported penalty types.
solverPenalty = {'saga':['l1','l2','none'],
                 'newton-cg':['l2', 'none'], 
                 'sag':['l2','none'],
                 'lbfgs':['l2','none']}
# List of regularization constants
regularization_constant = [0.0001,0.01,1,100,1e5]
modelDict = {}
```

### Creating functions that will store the confusion matrix values.


```python
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
```

### Creating a scoring dictionary that will contain model scoring methods from sklearn metrics library


```python
#Creating a scoring dictionary containing scoring methods from sklearn metric library
scoring = {'accuracy' : make_scorer(accuracy_score), \
           'precision' : make_scorer(precision_score), \
           'recall' : make_scorer(recall_score), \
           'f1_score' : make_scorer(f1_score),
           'tn': make_scorer(tn),\
           'fp': make_scorer(fp),\
           'fn': make_scorer(fn),\
           'tp': make_scorer(tp)}
```

### Creating All Models With Different Optimization Algorithms


```python
for solver in solverPenalty:
    for penalty in solverPenalty[solver]:
        for C in regularization_constant:
        # we create an instance of the Classifier
        # Logistic Regression (aka logit, MaxEnt) classifier.
            clf = linear_model.LogisticRegression(penalty=penalty,C=C,multi_class='multinomial',solver =solver)
            clf = clf.fit(X_train, y_train)
            #Performing cross-valiation with 10 K-folds
            scores = cross_validate(clf, X_train, y_train, cv=10, scoring=scoring, return_train_score =True)
            string = 'Solver: '+solver+', Penalty: '+penalty + ' Regularization Constant '+str(C)
            performanceFalseNegatives[string] = scores['test_fn'].mean()
            performanceF1Score[string] = scores['test_f1_score'].mean()
            modelParams[string] = {'Accuracy':scores['test_accuracy'].mean(),\
                    'Precision':scores['test_precision'].mean(),'Recall':scores['test_recall'].mean(),
                    'F-Score':scores['test_f1_score'].mean(), 'False Negatives' : scores['test_fn'].mean(),
                    'False Positives':scores['test_fp'].mean(),'True Positives':scores['test_tp'].mean(),
                    'True Negatives':scores['test_tn'].mean(),'Train Scores':scores['train_f1_score']}
            modelDict[string] = clf
```

### Selecting the Logistic Regression Model


```python
#print('Best Optimzation Algorithm as per F1 Score is: ',max(performanceF1Score.items(), key=operator.itemgetter(1))[0])
bestModelFScore = modelDict[max(performanceF1Score.items(), key=operator.itemgetter(1))[0]]
print('The model with best F1 Score is: ',bestModelFScore,'\n',\
    'and with a mean Accuracy of:',format(modelParams[max(performanceF1Score.items(), key=operator.itemgetter(1))[0]]['Accuracy'],'.2f'))#,'\n',\
#      'Precision:',format(modelParams[bestModelFScore]['Precision'],'.2f'),'\n',\
#      'Recall:',format(modelParams[bestModelFScore]['Recall'],'.2f'),'\n',\
#      'F-Score:',format(modelParams[bestModelFScore]['F-Score'],'.2f'),'\n',\
#      'False Negatives:',format(modelParams[bestModelFScore]['False Negatives'],'.2f'))
y_true, y_pred = y_test, bestModelFScore.predict(X_test)
print('\n','Classification Report','\n')
print(classification_report(y_true, y_pred))
print('Confusion Matrix','\n')
print(confusion_matrix(y_true, y_pred))
print("\n","Estimated coefficients for the linear regression","\n")
print(bestModelFScore.coef_ )

```

    The model with best F1 Score is:  LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='multinomial', n_jobs=None, penalty='l1',
                       random_state=None, solver='saga', tol=0.0001, verbose=0,
                       warm_start=False) 
     and with a mean Accuracy of: 0.98
    
     Classification Report 
    
                  precision    recall  f1-score   support
    
               0       0.99      0.96      0.97       117
               1       0.91      0.98      0.95        54
    
        accuracy                           0.96       171
       macro avg       0.95      0.97      0.96       171
    weighted avg       0.97      0.96      0.97       171
    
    Confusion Matrix 
    
    [[112   5]
     [  1  53]]
    
     Estimated coefficients for the linear regression 
    
    [[ 0.19375369  0.08838819  0.16665776  0.21262302  0.         -0.01322495
       0.26485832  0.31169401  0.         -0.07924885  0.63076619  0.
       0.27112121  0.36474721  0.         -0.1568497   0.          0.
      -0.1332635  -0.20444131  0.5940071   0.53784273  0.46621634  0.50519008
       0.42286827  0.          0.34366053  0.303753    0.21668217  0.        ]]


In logistic regression 'C' handles the extent to which the model regularizes, or prevents overfitting.
Here C is the inverse regularization constant i.e lower values of C indicate more regularization and less overfitting.
Hence in this case, the Model with best F1 score has a good amount of regularization @ C value = 1.

The model with the "saga" solver gives the best performance. It has high value of precision and recall and f-score for the minority cases "Malign", which is what we expect from a model in this case. Additionally the model predicted correctly 53 out of 54 cases of "Malignant" tumours in the testing dataset and wrongly predicted 5 out 117 cases as "Malignant" when it was "Benign" and only one case as "Benign" when it was truly "Malignant". Moreover, using the "l1" penalty the model has done automatic feature selection and eliminated 8 features.

#### Creating ROC Curve


```python
# Probabilites
y_prob = bestModelFScore.predict_proba(X_test)
prob = y_prob[:,1:]
fpr, tpr,thresholds = metrics.roc_curve(y_true,prob, drop_intermediate=False )
plt.figure(figsize=(22,9))
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

plt.figure(figsize=(22,9))
scikitplot.metrics.plot_lift_curve(y_true,y_prob,title='Lift Curve',figsize=(22,9), title_fontsize='large',text_fontsize="large")
plt.show()
```


![png](output_71_0.png)



    <Figure size 1584x648 with 0 Axes>



![png](output_71_2.png)

The ROC Curve plots the true positive rate vs the False Positive Rate. For the above model we can see that the curve rises vertically then towards the right which is a good indicator. Any curve which lies above the diagonal predicts better than a model that guesses randomly.

The lift curve tells us, that by how much the model is able to predict a class better than a model that guesses randomly. We can see above that the model consistently performs better than a random model.

### Decision Tree

#### Creating visualizations as to how different parameters affect the F-score of a Decision Tree


```python
## Dictionary that holds different parameters and their values.
parameters_dict = {"max_depth": range(2,40), "min_samples_split" : np.arange(0.05,0.4,0.05), \
                   "min_samples_leaf" : np.arange(0.05,0.5,0.05), "criterion": ["gini","entropy"], "min_impurity_decrease":np.arange(0.05,0.5,.05)} 
train_scores, test_scores = validation_curve(
    tree.DecisionTreeClassifier(class_weight='balanced'), X_train, y_train, param_name="max_depth", cv=10, 
    param_range=parameters_dict['max_depth'],scoring="f1")

train_scoresRecall, test_scoresRecall = validation_curve(
    tree.DecisionTreeClassifier(class_weight='balanced'), X_train, y_train, param_name="min_samples_split", cv=10, 
    param_range=parameters_dict['min_samples_split'],scoring="f1")

train_scoresLeaf, test_scoresLeaf = validation_curve(
    tree.DecisionTreeClassifier(class_weight='balanced'), X_train, y_train, param_name="min_samples_leaf", cv=10, 
    param_range=parameters_dict['min_samples_leaf'],scoring="f1")

train_scoresreduc, test_scoresreduc = validation_curve(
    tree.DecisionTreeClassifier(class_weight='balanced'), X_train, y_train, param_name="min_impurity_decrease", cv=10, 
    param_range=parameters_dict['min_impurity_decrease'],scoring="f1")


#Calculating mean and standard deviations of the scores
meanTrainScore = np.mean(train_scores, axis =1)
stdDevTrain = np.std(train_scores, axis=1)
meanTestScore = np.mean(test_scores, axis=1)
stdTestScore = np.std(test_scores, axis=1)

meanTrainScoreRC = np.mean(train_scoresRecall, axis =1)
stdDevTrainRC = np.std(train_scoresRecall, axis=1)
meanTestScoreRC = np.mean(test_scoresRecall, axis=1)
stdTestScoreRC = np.std(test_scoresRecall, axis=1)

meanTrainScoreLeaf = np.mean(train_scoresLeaf, axis =1)
stdDevTrainLeaf = np.std(train_scoresLeaf, axis=1)
meanTestScoreLeaf = np.mean(test_scoresLeaf, axis=1)
stdTestScoreLeaf = np.std(test_scoresLeaf, axis=1)

meanTrainScorereduc = np.mean(train_scoresreduc, axis =1)
stdDevTrainreduc = np.std(train_scoresreduc, axis=1)
meanTestScorereduc = np.mean(test_scoresreduc, axis=1)
stdTestScorereduc = np.std(test_scoresreduc, axis=1)

#Plotting 
plt.figure(figsize=(22,9))

plt.title("Test-Train Recall Varying by max_depth")
plt.xlabel("max_depth")
plt.ylabel("F-Score")
plt.ylim(0.8, 1.05)
plt.fill_between(parameters_dict['max_depth'], meanTrainScore - stdDevTrain, meanTrainScore + stdDevTrain, alpha=0.2, color="r")
plt.plot(parameters_dict['max_depth'], meanTrainScore, label="Training score (Max Depth)",
             color="b")
plt.fill_between(parameters_dict['max_depth'], meanTestScore - stdTestScore, meanTestScore + stdTestScore, alpha=0.2, color="g")
plt.plot(parameters_dict['max_depth'], meanTestScore, label="Cross-validation score (Max Depth)",
             color="r")

plt.legend(loc="best")
plt.xticks(parameters_dict['max_depth'])
plt.show()

plt.figure(figsize=(22,9))
plt.title("Test-Train Recall Varying by min_sample_split")
plt.xlabel("min_sample_split")
plt.ylabel("F-Score")
plt.ylim(0.8, 1.05)
plt.fill_between(parameters_dict['min_samples_split'], meanTrainScoreRC - stdDevTrainRC, meanTrainScoreRC + stdDevTrainRC, alpha=0.2, color="r")
plt.plot(parameters_dict['min_samples_split'], meanTrainScoreRC, label="Training score (Min Sample Split)",
             color="g")
plt.fill_between(parameters_dict['min_samples_split'], meanTestScoreRC - stdTestScoreRC, meanTestScoreRC + stdTestScoreRC, alpha=0.2, color="g")
plt.plot(parameters_dict['min_samples_split'], meanTestScoreRC, label="Cross-validation score (Min Sample Split)",
             color="y")

plt.legend(loc="best")
plt.xticks(parameters_dict['min_samples_split'])
plt.show()

plt.figure(figsize=(22,9))
plt.title("Test-Train Recall Varying by min_sample_leaf")
plt.xlabel("min_sample_leaf")
plt.ylabel("F-Score")
plt.ylim(0.8, 1.0)
plt.fill_between(parameters_dict['min_samples_leaf'], meanTrainScoreLeaf - stdDevTrainLeaf, meanTrainScoreLeaf + stdDevTrainLeaf, alpha=0.2, color="r")
plt.plot(parameters_dict['min_samples_leaf'], meanTrainScoreLeaf, label="Training score (Min Sample Leaf)",
             color="g")
plt.fill_between(parameters_dict['min_samples_leaf'], meanTestScoreLeaf - stdTestScoreLeaf, meanTestScoreLeaf + stdTestScoreLeaf, alpha=0.2, color="g")
plt.plot(parameters_dict['min_samples_leaf'], meanTestScoreLeaf, label="Cross-validation score (Min Sample Leaf)",
             color="y")

plt.legend(loc="best")
plt.xticks(parameters_dict['min_samples_leaf'])
plt.show()
plt.figure(figsize=(22,9))
plt.title("Test-Train Recall Varying by min_impurity_decrease")
plt.xlabel("min_impurity_decrease")
plt.ylabel("F-Score")
plt.ylim(0.1, 1.1)
plt.fill_between(parameters_dict['min_impurity_decrease'], meanTrainScorereduc - stdDevTrainreduc, meanTrainScorereduc + stdDevTrainreduc, alpha=0.2, color="r")
plt.plot(parameters_dict['min_impurity_decrease'], meanTrainScorereduc, label="Training score (min_impurity_decrease)",
             color="b")
plt.fill_between(parameters_dict['min_impurity_decrease'], meanTestScorereduc - stdTestScorereduc, meanTestScorereduc + stdTestScorereduc, alpha=0.2, color="g")
plt.plot(parameters_dict['min_impurity_decrease'], meanTestScorereduc, label="Cross-validation score (min_impurity_decrease)",
             color="g")

plt.legend(loc="best")
plt.xticks(parameters_dict['min_impurity_decrease'])


plt.show()


```


![png](output_74_0.png)



![png](output_74_1.png)



![png](output_74_2.png)



![png](output_74_3.png)


From the above graphs it can be seen how each individual hyperparameter effects the F-Score of the model. From the **Test-Train F-Score Varying by max_depth graph**, it can be seen, that after max_depth =2 , the testing F-score increases upto when max_depth =11, post which there is a jagged pattern. But there is no significant drop in the test recall levels as max_depth increases. But to keep the model simple I would choose the values of 5-11. 

Looking at the **Test-Train F-Score Varying by min_sample_split** There seems no underfitting or overfitting at min_sample_split as the value of the hyperparameter increases. But we see a drop in performance as the values increase from 0.05. I'm going to investigate this hyperparater in the range of 0.05 to 0.1.

Looking at the **Test-Train F-Score Varying by min_sample_leaf**,there is hardly any variation in the performance across the values of the hyperparamter. Hence im going to arbitarily choose a range of 0.05 to 0.15.

**Test-Train F-Score Varying by min_impurity_decrease** is not very helpful in this case, the train and test follow very closely and performance drops hugely after 0.35, means the model is not able to decrease the impurity by that amount at any node, hence its better to use it as its default value.

Also, I would be trying the default values of all the above parameters.

### Creating the model as per above hyper-parameter analysis and displaying metrics of the model


```python
treeParamsTuned = {'max_depth':range(5,11,1), 'min_samples_split':[2,0.05,0.1], 'min_samples_leaf':[1,2,0.05,0.1]}

clfTree = GridSearchCV(tree.DecisionTreeClassifier(), treeParamsTuned, cv=10,scoring="f1")
clfTree.fit(X_train, y_train)
print('The best parameter values are ',clfTree.best_params_,'\n')
print("Detailed classification report:")
print()
y_true, y_pred = y_test, clfTree.best_estimator_.predict(X_test)
print(classification_report(y_true, y_pred))
print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))
print()
print("Feature Importance (Esitmate of total reduction in entropy brought by a feature)")
print(clfTree.best_estimator_.feature_importances_)
```

    The best parameter values are  {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2} 
    
    Detailed classification report:
    
                  precision    recall  f1-score   support
    
               0       0.97      0.93      0.95       117
               1       0.86      0.94      0.90        54
    
        accuracy                           0.94       171
       macro avg       0.92      0.94      0.93       171
    weighted avg       0.94      0.94      0.94       171
    
    Confusion Matrix
    [[109   8]
     [  3  51]]
    
    Feature Importance (Esitmate of total reduction in entropy brought by a feature)
    [0.         0.04644171 0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.03129045 0.         0.         0.         0.
     0.         0.         0.         0.00077867 0.         0.77622998
     0.01626736 0.02021388 0.01437431 0.09440364 0.         0.        ]


    C:\Users\maina\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)


The decision tree has also performed very good, and has high values of precision, recall and f-score for the minority case "Malign". Additionally the model predicted correctly 51 out of 54 cases of "Malignant" tumours in the testing dataset and wrongly predicted 10 out 117 cases as "Malignant" when it was "Benign" and only 3 cases as "Benign" when it was truly "Malignant". Also looking at the feature set and their respective estimates of total entropy reduction, the model has eliminated 19 features.

#### Creating ROC and Lift Curves


```python
# Probabilites
y_prob = clfTree.best_estimator_.predict_proba(X_test)
prob = y_prob[:,1:]
fpr, tpr,thresholds = metrics.roc_curve(y_true,prob, drop_intermediate=False )
plt.figure(figsize=(22,9))
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

plt.figure(figsize=(22,9))
scikitplot.metrics.plot_lift_curve(y_true,y_prob,title='Lift Curve',figsize=(22,9), title_fontsize='large',text_fontsize="large")
plt.show()
```


![png](output_80_0.png)



    <Figure size 1584x648 with 0 Axes>



![png](output_80_2.png)

The ROC Curve plots the true positive rate vs the False Positive Rate. For the above model we can see that the curve rises vertically then towards the right which is a good indicator but not as steep as logistic regression. Any curve which lies above the diagonal predicts better than a model that guesses randomly.

The lift curve tells us, that by how much the model is able to predict a class better than a model that guesses randomly. We can see above that the model consistently performs better than a random model.

### Creating the plot to see the effect of 'n' on the KNN model for the dataset.


```python
# Create range of values for parameter
neighbor = range(1,25,1)

train_scores, test_scores = validation_curve(
    neighbors.KNeighborsClassifier(), X_train, y_train, param_name="n_neighbors", cv=10, 
    param_range=neighbor,
    scoring="f1")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(22,9))
plt.title("Validation Curve with K-NN Neighbors")
plt.xlabel("k neighbors")
plt.ylabel("F-Score")
plt.ylim(0.8, 1.1)

# Plot the values for recall with K ranging from 1-20
plt.plot(neighbor, train_scores_mean, label="Training score",
             color="r")
plt.fill_between(neighbor, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="b")

plt.plot(neighbor, test_scores_mean, label="Cross-validation score",
             color="g")
plt.fill_between(neighbor, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="r")
plt.legend(loc="best")
plt.xticks(neighbor)

plt.show()
```


![png](output_82_0.png)


From the above graph it seems at n = 1, we can see that the model as completely memorized the dataset and has a f-score of 1, and the test scores is less (largest gap). The model is overfitting here. As we increase the n value we see that the test and train performance is almost the same, indicating that its not over or underfitting.

### Finding the best value of n for the KNN Classifier and displaying the results of the best value.


```python
KNNParamsTuned = {'n_neighbors':range(1,25,1)}#, 'min_samples_split':np.arange(0.05,0.3,0.05), 'min_samples_leaf':np.arange(0.05,0.3,0.05), 'min_impurity_decrease':np.arange(0.05,0.3,0.05)}
#X_train, X_test, y_train, y_test = train_test_split(dataFile.iloc[:,2:], dataFile[1], test_size=0.33, random_state=42)
clfKNN = GridSearchCV(neighbors.KNeighborsClassifier(), KNNParamsTuned, cv=10,scoring="f1", return_train_score =True)
clfKNN.fit(X_train, y_train)
print('The best parameter values are ',clfKNN.best_params_,'\n')
print("Detailed classification report:")
print()
y_true, y_pred = y_test, clfKNN.best_estimator_.predict(X_test)
print(classification_report(y_true, y_pred))
print('Confusion Matrix','\n')
print(confusion_matrix(y_true, y_pred))
```

    The best parameter values are  {'n_neighbors': 5} 
    
    Detailed classification report:
    
                  precision    recall  f1-score   support
    
               0       0.98      0.96      0.97       117
               1       0.91      0.96      0.94        54
    
        accuracy                           0.96       171
       macro avg       0.95      0.96      0.95       171
    weighted avg       0.96      0.96      0.96       171
    
    Confusion Matrix 
    
    [[112   5]
     [  2  52]]


    C:\Users\maina\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)


From the above results we can see that the KNN Model with n = 5, provides good performance. The recall and f-score for both positive and negative class is 0.96 and Additionally the model predicted correctly 52 out of 54 cases of "Malignant" tumours in the testing dataset and wrongly predicted 5 out 117 cases as "Malignant" when it was "Benign" and only two cases as "Benign" when it was truly "Malignant".

#### Creating ROC and Lift Curves


```python
# Probabilites
y_prob = clfKNN.best_estimator_.predict_proba(X_test)
prob = y_prob[:,1:]
fpr, tpr,thresholds = metrics.roc_curve(y_true,prob, drop_intermediate=False )
plt.figure(figsize=(22,9))
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

plt.figure(figsize=(22,9))
scikitplot.metrics.plot_lift_curve(y_true,y_prob,title='Lift Curve',figsize=(22,9), title_fontsize='large',text_fontsize="large")
plt.show()
```


![png](output_88_0.png)



    <Figure size 1584x648 with 0 Axes>



![png](output_88_2.png)

The ROC Curve plots the true positive rate vs the False Positive Rate. For the above model we can see that the curve rises vertically then towards the right which is a good indicator. Any curve which lies above the diagonal predicts better than a model that guesses randomly.

The lift curve tells us, that by how much the model is able to predict a class better than a model that guesses randomly. We can see above that the model consistently performs better than a random model.

### SVM

#### Visualizing effect of hyperparameters on SVM


```python
parameters_dict={"gamma":[0.01, 1.0,10,100]}
```


```python
train_scores, test_scores = validation_curve(
    svm.SVC(kernel="linear"),X_train, y_train, param_name="gamma", cv=5, 
    param_range=parameters_dict['gamma'],scoring="f1")

#Calculating mean and standard deviations of the scores
meanTrainScore = np.mean(train_scores, axis =1)
stdDevTrain = np.std(train_scores, axis=1)
meanTestScore = np.mean(test_scores, axis=1)
stdTestScore = np.std(test_scores, axis=1)

#Plotting 
plt.figure(figsize=(22,9))
#plt.subplot(2,2,1)
plt.title("Test-Train F1-Score Varying by gamma for Linear Kernel")
plt.xlabel("gamma")
plt.ylabel("F1-Score")
plt.ylim(0.9, 1.05)
plt.fill_between(parameters_dict['gamma'], meanTrainScore - stdDevTrain, meanTrainScore + stdDevTrain, alpha=0.2, color="r")
plt.plot(parameters_dict['gamma'], meanTrainScore, label="Training score (gamma)",
             color="b")
plt.fill_between(parameters_dict['gamma'], meanTestScore - stdTestScore, meanTestScore + stdTestScore, alpha=0.2, color="g")
plt.plot(parameters_dict['gamma'], meanTestScore, label="Cross-validation score (gamma)",
             color="r")

plt.legend(loc="best")
plt.xticks(parameters_dict['gamma'])
plt.show()

#########################################################################################################

train_scores, test_scores = validation_curve(
    svm.SVC(kernel="rbf"),X_train, y_train, param_name="gamma", cv=5, 
    param_range=parameters_dict['gamma'],scoring="f1")

#Calculating mean and standard deviations of the scores
meanTrainScore = np.mean(train_scores, axis =1)
stdDevTrain = np.std(train_scores, axis=1)
meanTestScore = np.mean(test_scores, axis=1)
stdTestScore = np.std(test_scores, axis=1)

#Plotting 
plt.figure(figsize=(22,9))
#plt.subplot(2,2,1)
plt.title("Test-Train F1-Score Varying by gamma for Radial Basis Function Kernel")
plt.xlabel("gamma")
plt.ylabel("F1-Score")
plt.ylim(0.4, 1.05)
plt.fill_between(parameters_dict['gamma'], meanTrainScore - stdDevTrain, meanTrainScore + stdDevTrain, alpha=0.2, color="r")
plt.plot(parameters_dict['gamma'], meanTrainScore, label="Training score (gamma)",
             color="b")
plt.fill_between(parameters_dict['gamma'], meanTestScore - stdTestScore, meanTestScore + stdTestScore, alpha=0.2, color="g")
plt.plot(parameters_dict['gamma'], meanTestScore, label="Cross-validation score (gamma)",
             color="r")

plt.legend(loc="best")
plt.xticks(parameters_dict['gamma'])
plt.show()

#########################################################################################################

train_scores, test_scores = validation_curve(
    svm.SVC(kernel="poly"),X_train, y_train, param_name="gamma", cv=5, 
    param_range=parameters_dict['gamma'],scoring="f1")

#Calculating mean and standard deviations of the scores
meanTrainScore = np.mean(train_scores, axis =1)
stdDevTrain = np.std(train_scores, axis=1)
meanTestScore = np.mean(test_scores, axis=1)
stdTestScore = np.std(test_scores, axis=1)

#Plotting 
plt.figure(figsize=(22,9))
#plt.subplot(2,2,1)
plt.title("Test-Train F1-Score Varying by gamma for Polynomial Kernel")
plt.xlabel("gamma")
plt.ylabel("F1-Score")
plt.ylim(0.6, 1.05)
plt.fill_between(parameters_dict['gamma'], meanTrainScore - stdDevTrain, meanTrainScore + stdDevTrain, alpha=0.2, color="r")
plt.plot(parameters_dict['gamma'], meanTrainScore, label="Training score (gamma)",
             color="b")
plt.fill_between(parameters_dict['gamma'], meanTestScore - stdTestScore, meanTestScore + stdTestScore, alpha=0.2, color="g")
plt.plot(parameters_dict['gamma'], meanTestScore, label="Cross-validation score (gamma)",
             color="r")

plt.legend(loc="best")
plt.xticks(parameters_dict['gamma'])
plt.show()
```


![png](output_92_0.png)



![png](output_92_1.png)



![png](output_92_2.png)


### Finding the best model for SVM


```python
parameters_dict={"gamma":[0.01, 1.0,10,100],"kernel":["linear","rbf","poly"]}
#X_train, X_test, y_train, y_test = train_test_split(dataFile.iloc[:,2:], dataFile[1], test_size=0.33, random_state=42)
svmm = GridSearchCV(svm.SVC(probability=True), parameters_dict, cv=10,scoring="f1", return_train_score =True)
svmm.fit(X_train, y_train)
print('The best parameter values are ',svmm.best_params_,'\n')
print("Detailed classification report:")
print()
y_true, y_pred = y_test, svmm.best_estimator_.predict(X_test)
print(classification_report(y_true, y_pred))
print('Confusion Matrix','\n')
print(confusion_matrix(y_true, y_pred))
```

    The best parameter values are  {'gamma': 0.01, 'kernel': 'linear'} 
    
    Detailed classification report:
    
                  precision    recall  f1-score   support
    
               0       1.00      0.94      0.97       117
               1       0.89      1.00      0.94        54
    
        accuracy                           0.96       171
       macro avg       0.94      0.97      0.95       171
    weighted avg       0.96      0.96      0.96       171
    
    Confusion Matrix 
    
    [[110   7]
     [  0  54]]


    C:\Users\maina\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)

```python
#### Creating ROC and Lift Curves
```


```python
# Probabilites
y_prob = svmm.best_estimator_.predict_proba(X_test)
prob = y_prob[:,1:]
fpr, tpr,thresholds = metrics.roc_curve(y_true,prob, drop_intermediate=False )
plt.figure(figsize=(22,9))
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()
plt.figure(figsize=(22,9))
scikitplot.metrics.plot_lift_curve(y_true,y_prob,title='Lift Curve',figsize=(22,9), title_fontsize='large',text_fontsize="large")
plt.show()
```


![png](output_96_0.png)



    <Figure size 1584x648 with 0 Axes>



![png](output_96_2.png)

The ROC Curve plots the true positive rate vs the False Positive Rate. For the above model we can see that the curve rises vertically then towards the right which is perfect. Any curve which lies above the diagonal predicts better than a model that guesses randomly.

The lift curve tells us, that by how much the model is able to predict a class better than a model that guesses randomly. We can see above that the model consistently performs better than a random model.

### 5) [25 points] [Mining publicly available data] Download the dataset on car evaluations from http://archive.ics.uci.edu/ml/datasets/Car+Evaluation (this link also has the description of the data). This dataset has 1728 records, each record representing a car evaluation. Each car evaluation is described with 7 attributes. 6 of the attributes represent car characteristics, such as buying price, price of the maintenance, number of doors, capacity in terms of persons to carry, the size of luggage boot, and estimated safety of the car. The seventh variable represents the evaluation of the car (unacceptable, acceptable, good, very good).

Your task: Among the basic classification techniques that you are familiar with (i.e., decision tree, k-NN, logistic regression, NB, SVM) use all that would be applicable to this dataset to predict the evaluation of the cars based on their characteristics. Explore how well these techniques perform for several different parameter values. Present a brief overview of your predictive modeling process, explorations, and discuss your results. Present your final model (i.e., the best predictive model that you were able to come up with), and discuss its performance in a comprehensive manner (overall accuracy; per-class performance, i.e., whether this model predicts all classes equally well, or if there some classes for which it does much better than others; etc.).

### Approach

Looking at the data the Good and Very Good class has very few samples in the data, hence to make sure that the training and testing dataset has all type of samples, I would be using Stratified Sampling.
For model hyperparameter tuning and model performance evaluation I would be using nested cross-validation. 
To mimic categorical variables I used one-hot encoding and for the continuous dataset i would be replacing each category within each attribute with a number. The exact encoding is present in below sections. 

Pros and Cons of Converting Ordinal Variables to Categorical:
    Cons : Ordinal to Nominal: We lose information of order and the distance between each category is now the same.
    Pros : Its easier to algorithms to process.
One way to process Ordinal to Categorical is to do one-hot encoding. If the cardinality of the categorical features is low (relative to the amount of data) one-hot encoding will work best. We can use it as input into any model. But if the cardinality is large and our dataset is small, one-hot encoding may not be feasible, and the model may not be able to efficiently learn.

Pros and Cons of Converting Ordinal Variables to Numerical:
    Cons: There is a risk of overfitting.
    Pros: We can  assign different distances between the categories. Hence even order can be preserved.


#### Load the dataset and creating stratified folds


```python
carData = pd.read_csv('car.data',header = None, names = ['buying','maint','doors','persons','lug_boot','safety'], index_col = False)
target = pd.read_csv('car.data', header = None, names=['target'], usecols=[6], index_col = False, squeeze = True)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = 40)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = 40)
```

### ONE HOT ENCODING (CATEGORICAL)


```python
enc = preprocessing.OneHotEncoder(handle_unknown="ignore")
transformedData = pd.DataFrame(enc.fit_transform(carData).toarray())
transformedData.columns = enc.get_feature_names()
transformedData.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0_high</th>
      <th>x0_low</th>
      <th>x0_med</th>
      <th>x0_vhigh</th>
      <th>x1_high</th>
      <th>x1_low</th>
      <th>x1_med</th>
      <th>x1_vhigh</th>
      <th>x2_2</th>
      <th>x2_3</th>
      <th>...</th>
      <th>x2_5more</th>
      <th>x3_2</th>
      <th>x3_4</th>
      <th>x3_more</th>
      <th>x4_big</th>
      <th>x4_med</th>
      <th>x4_small</th>
      <th>x5_high</th>
      <th>x5_low</th>
      <th>x5_med</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>

### FOR CONTINUOUS VARIABLES


```python
import category_encoders
ordinal_cols_mapping = [{"col": "buying", 
  "mapping": {"vhigh":4, 
              "high":3, 
              "med":2,
              "low":1
}},{"col": "maint", 
  "mapping": {"vhigh":4, 
              "high":3, 
              "med":2,
              "low":1
}},{"col": "doors", 
  "mapping": {"2":2, 
              "3":3, 
              "4":4,
              "5more":5
}},{"col": "persons", 
  "mapping": {"2":2,  
              "4":4,
              "more":5
}},{"col": "lug_boot", 
  "mapping": {"small":1,  
              "med":2,
              "big":3
}},{"col": "safety", 
  "mapping": {"low":1,  
              "med":2,
              "high":3
}}]
encoder = category_encoders.OrdinalEncoder(mapping = ordinal_cols_mapping, return_df = True)
transformedContData = encoder.fit_transform(carData)
transformedContData.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buying</th>
      <th>maint</th>
      <th>doors</th>
      <th>persons</th>
      <th>lug_boot</th>
      <th>safety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



#### Creating arrays to hold scoring information per class


```python
precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []

def empty_arrays():
    precisionArrayAcc = []
    recallArrayAcc = []
    f1ArrayAcc = []

    precisionArrayGood = []
    recallArrayGood = []
    f1ArrayGood = []

    precisionArrayUnacc = []
    recallArrayUnacc = []
    f1ArrayUnacc = []

    precisionArrayVgood = []
    recallArrayVgood = []
    f1ArrayVgood = []
```

#### Creating a scoring function that will hold per class information


```python
def scoringFunction(y_pred,y_true):
#    print(precision_score(y_pred,y_true, average = None))
    pScores = precision_score(y_pred,y_true, average = None)
    rScores = recall_score(y_pred,y_true, average = None)
    fScores = f1_score(y_pred,y_true, average = None)
    
    precisionArrayAcc.append(pScores[0]) 
    precisionArrayGood.append(pScores[1])
    precisionArrayUnacc.append(pScores[2])
    precisionArrayVgood.append(pScores[3])
    
    recallArrayAcc.append(rScores[0]) 
    recallArrayGood.append(rScores[1])
    recallArrayUnacc.append(rScores[2])
    recallArrayVgood.append(rScores[3])
    
    f1ArrayAcc.append(fScores[0]) 
    f1ArrayGood.append(fScores[1])
    f1ArrayUnacc.append(fScores[2])
    f1ArrayVgood.append(fScores[3])
    return accuracy_score(y_pred,y_true)
```

#### Creating a function that will hold mean and standard deviation of scores for nested cross validation.


```python
data = {}
def createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood):
    data = {"Mean Precision":[np.array(precisionArrayAcc).mean(),\
                          np.array(precisionArrayGood).mean(),\
                          np.array(precisionArrayUnacc).mean(),\
                          np.array(precisionArrayVgood).mean()],\
        "Standard Deviation Precision":[np.array(precisionArrayAcc).std(),\
                          np.array(precisionArrayGood).std(),\
                          np.array(precisionArrayUnacc).std(),\
                          np.array(precisionArrayVgood).std()],\
        "Mean Recall":[np.array(recallArrayAcc).mean(),\
                          np.array(recallArrayGood).mean(),\
                          np.array(recallArrayUnacc).mean(),\
                          np.array(recallArrayVgood).mean()],\
        "Standard Deviation Recall":[np.array(recallArrayAcc).std(),\
                          np.array(recallArrayGood).std(),\
                          np.array(recallArrayUnacc).std(),\
                          np.array(recallArrayVgood).std()],\
        "Mean F1":[np.array(f1ArrayAcc).mean(),\
                          np.array(f1ArrayGood).mean(),\
                          np.array(f1ArrayUnacc).mean(),\
                          np.array(f1ArrayVgood).mean()],\
        "Standard Deviation F1":[np.array(f1ArrayAcc).std(),\
                          np.array(f1ArrayGood).std(),\
                          np.array(f1ArrayUnacc).std(),\
                          np.array(f1ArrayVgood).std()]}
    return pd.DataFrame(data, index=["Acceptable","Good","Unacceptable","Very Good"])
```

### Decision Tree

### Creating visualizations as to how different parameters affect the Accuracy of a Decision Tree


```python
## Dictionary that holds different parameters and their values.
parameters_dict = {"max_depth": range(2,20), "min_samples_split" : np.arange(2,10,2), \
                   "min_samples_leaf" : np.arange(1,10,2), "criterion": ["gini","entropy"], "min_impurity_decrease":np.arange(0.05,0.5,.05)}
def treeValidationCurve(tDatam, tar):
    train_scores, test_scores = validation_curve(
        tree.DecisionTreeClassifier(),tDatam, tar, param_name="max_depth", cv=5, 
        param_range=parameters_dict['max_depth'],scoring="accuracy")

    train_scoresRecall, test_scoresRecall = validation_curve(
        tree.DecisionTreeClassifier(),tDatam, tar, param_name="min_samples_split", cv=5, 
        param_range=parameters_dict['min_samples_split'],scoring="accuracy")

    train_scoresLeaf, test_scoresLeaf = validation_curve(
        tree.DecisionTreeClassifier(), tDatam, tar, param_name="min_samples_leaf", cv=5, 
        param_range=parameters_dict['min_samples_leaf'],scoring="accuracy")

    train_scoresreduc, test_scoresreduc = validation_curve(
        tree.DecisionTreeClassifier(),tDatam, tar, param_name="min_impurity_decrease", cv=5, 
        param_range=parameters_dict['min_impurity_decrease'],scoring="accuracy")


    #Calculating mean and standard deviations of the scores
    meanTrainScore = np.mean(train_scores, axis =1)
    stdDevTrain = np.std(train_scores, axis=1)
    meanTestScore = np.mean(test_scores, axis=1)
    stdTestScore = np.std(test_scores, axis=1)


    #Plotting 
    plt.figure(figsize=(22,9))
    #plt.subplot(2,2,1)
    plt.title("Test-Train Accuracy Varying by max_depth")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.ylim(0.6, 1.05)
    plt.fill_between(parameters_dict['max_depth'], meanTrainScore - stdDevTrain, meanTrainScore + stdDevTrain, alpha=0.2, color="r")
    plt.plot(parameters_dict['max_depth'], meanTrainScore, label="Training score (Max Depth)",
                 color="b")
    plt.fill_between(parameters_dict['max_depth'], meanTestScore - stdTestScore, meanTestScore + stdTestScore, alpha=0.2, color="g")
    plt.plot(parameters_dict['max_depth'], meanTestScore, label="Cross-validation score (Max Depth)",
                 color="r")

    plt.legend(loc="best")
    plt.xticks(parameters_dict['max_depth'])
    plt.show()
treeValidationCurve(transformedData, target)
```


![png](output_114_0.png)


### Decision Tree With Continuous Data

### Creating visualizations as to how different parameters affect the Accuracy of a Decision Tree


```python
parameters_dict = {"max_depth": range(2,20), "min_samples_split" : np.arange(2,10,2), \
                   "min_samples_leaf" : np.arange(1,10,2), "criterion": ["gini","entropy"], "min_impurity_decrease":np.arange(0.05,0.5,.05)}

treeValidationCurve(transformedContData, target)
```


![png](output_117_0.png)

The model behaves almost similarly on both the datasets, but the model trained on numerical data seems to have a larger standard deviation.

#### Cross Validation On Data Categorical Data


```python
precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []

parameters_dict = {"max_depth": range(3,10), "min_samples_split" : np.arange(2,10,2), \
                   "min_samples_leaf" : np.arange(2,10,2), "criterion": ["gini"]}

clfTree = GridSearchCV(tree.DecisionTreeClassifier(), parameters_dict, cv=inner_cv, scoring="accuracy", refit=True)
nested_score = cross_validate(clfTree, X=transformedData, y=target, cv=outer_cv, scoring = make_scorer(scoringFunction))

print("Mean Accuracy Categorical: {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)


```

    Mean Accuracy Categorical: 0.95, Std Deviation: 0.01

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Standard Deviation Precision</th>
      <th>Mean Recall</th>
      <th>Standard Deviation Recall</th>
      <th>Mean F1</th>
      <th>Standard Deviation F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.881290</td>
      <td>0.019193</td>
      <td>0.929768</td>
      <td>0.038091</td>
      <td>0.904736</td>
      <td>0.027157</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>0.820962</td>
      <td>0.137152</td>
      <td>0.810989</td>
      <td>0.098080</td>
      <td>0.802655</td>
      <td>0.066959</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>0.987425</td>
      <td>0.008303</td>
      <td>0.966942</td>
      <td>0.005844</td>
      <td>0.977043</td>
      <td>0.004338</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>0.856613</td>
      <td>0.076303</td>
      <td>0.876923</td>
      <td>0.115128</td>
      <td>0.861321</td>
      <td>0.068384</td>
    </tr>
  </tbody>
</table>
</div>



#### Cross Validation On Continuous Data


```python
precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []

nested_score = cross_validate(clfTree, X=transformedContData, y=target, cv=outer_cv, scoring = make_scorer(scoringFunction))

print("Mean Accuracy Continuous: {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)
```

    Mean Accuracy Continuous: 0.97, Std Deviation: 0.01

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Standard Deviation Precision</th>
      <th>Mean Recall</th>
      <th>Standard Deviation Recall</th>
      <th>Mean F1</th>
      <th>Standard Deviation F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.931573</td>
      <td>0.037101</td>
      <td>0.953213</td>
      <td>0.048984</td>
      <td>0.940669</td>
      <td>0.021211</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>0.862160</td>
      <td>0.074811</td>
      <td>0.812088</td>
      <td>0.071673</td>
      <td>0.830694</td>
      <td>0.028967</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>0.992712</td>
      <td>0.012590</td>
      <td>0.985950</td>
      <td>0.006714</td>
      <td>0.989250</td>
      <td>0.005813</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>0.910513</td>
      <td>0.053420</td>
      <td>0.923077</td>
      <td>0.068802</td>
      <td>0.915560</td>
      <td>0.051841</td>
    </tr>
  </tbody>
</table>
</div>

#### Decision Tree performance on both type of datasets is almost similar, only that the model performs marginally better than on the continuous data set. Both the models predict the "Unacceptable" class the best, followed by Acceptable, Very Good and Good, but the models have very similar recall for the "Good" class, that means they capture same amounts of true "Good" cases out of the total good cases.


```python
precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []
```

### Logistic Regression (Categorical Data)

#### Creating parameter dictionary


```python
parameters_dict = {"C":[0.000001, 0.0001, 0.001, 0.01, 1, 10000, 100000],"solver":["saga"], "penalty":["l1","l2"]  }
```

#### Nested Cross Validation For Logistic Regression


```python

logisticRegression = GridSearchCV(linear_model.LogisticRegression(multi_class="multinomial"), parameters_dict, cv=inner_cv, scoring="accuracy", refit=True)
nested_score = cross_validate(logisticRegression, X=transformedData, y=target.ravel(), cv=outer_cv, scoring = make_scorer(scoringFunction))
print("Mean Accuracy : {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)
```

    C:\Users\maina\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)


    Mean Accuracy : 0.93, Std Deviation: 0.01

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Standard Deviation Precision</th>
      <th>Mean Recall</th>
      <th>Standard Deviation Recall</th>
      <th>Mean F1</th>
      <th>Standard Deviation F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.893637</td>
      <td>0.050935</td>
      <td>0.912765</td>
      <td>0.056257</td>
      <td>0.902080</td>
      <td>0.043911</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>0.832093</td>
      <td>0.066163</td>
      <td>0.812637</td>
      <td>0.084496</td>
      <td>0.817675</td>
      <td>0.044546</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>0.980977</td>
      <td>0.015007</td>
      <td>0.974380</td>
      <td>0.014410</td>
      <td>0.977615</td>
      <td>0.012867</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>0.902239</td>
      <td>0.051269</td>
      <td>0.900000</td>
      <td>0.084615</td>
      <td>0.898765</td>
      <td>0.054640</td>
    </tr>
  </tbody>
</table>
</div>

### Logistic Regression (Continuous Data)


```python
precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []
parameters_dict = {"C":[0.000001, 0.0001, 0.001, 0.01, 1, 10000, 100000],"solver":["saga"], "penalty":["l1","l2"]  }
```


```python
logisticRegression = GridSearchCV(linear_model.LogisticRegression(multi_class="multinomial"), parameters_dict, cv=inner_cv, scoring="accuracy", refit=True)
nested_score = cross_validate(logisticRegression, X=transformedContData, y=target.ravel(), cv=outer_cv, scoring = make_scorer(scoringFunction))
print("Mean Accuracy : {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)
```

    Mean Accuracy : 0.84, Std Deviation: 0.02

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Standard Deviation Precision</th>
      <th>Mean Recall</th>
      <th>Standard Deviation Recall</th>
      <th>Mean F1</th>
      <th>Standard Deviation F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.695937</td>
      <td>0.030435</td>
      <td>0.666302</td>
      <td>0.073748</td>
      <td>0.679642</td>
      <td>0.052722</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>0.548209</td>
      <td>0.058199</td>
      <td>0.419780</td>
      <td>0.092046</td>
      <td>0.467191</td>
      <td>0.053740</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>0.905857</td>
      <td>0.018996</td>
      <td>0.934711</td>
      <td>0.015983</td>
      <td>0.919926</td>
      <td>0.014016</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>0.754444</td>
      <td>0.084357</td>
      <td>0.661538</td>
      <td>0.061538</td>
      <td>0.699442</td>
      <td>0.031513</td>
    </tr>
  </tbody>
</table>
</div>

#### There is a considerable difference between the performance difference between the predictive performance of logistic regression when using categorical and continuous data, and the categorical one outperforms the latter. The catergorical one has a mean Accuracy of 93% while the continuous one has a mean Accuracy of 84%, and the recall values are also very low for the continious dataset. Logistic regression is able to predict Unacceptable the best, followed by Very Good, Acceptable and then Good.

### Naive Baiyes (Categorical Data Only)

#### Parameter Dictionary


```python
parameters_dict={"alpha":[0.0,1.0,10.0,20.0,40.0,60.0,80.0,100.0]}
precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []
```

### Creating visualizations as to how different parameters affect the Accuracy of a Naive Baiyes


```python
train_scores, test_scores = validation_curve(
    naive_bayes.MultinomialNB(),transformedData, target, param_name="alpha", cv=5, 
    param_range=parameters_dict['alpha'],scoring="accuracy")

#Calculating mean and standard deviations of the scores
meanTrainScore = np.mean(train_scores, axis =1)
stdDevTrain = np.std(train_scores, axis=1)
meanTestScore = np.mean(test_scores, axis=1)
stdTestScore = np.std(test_scores, axis=1)

#Plotting 
plt.figure(figsize=(22,9))
#plt.subplot(2,2,1)
plt.title("Test-Train Accuracy Varying by alpha")
plt.xlabel("alpha")
plt.ylabel("Accuracy")
plt.ylim(0.6, 1.05)
plt.fill_between(parameters_dict['alpha'], meanTrainScore - stdDevTrain, meanTrainScore + stdDevTrain, alpha=0.2, color="r")
plt.plot(parameters_dict['alpha'], meanTrainScore, label="Training score (alpha)",
             color="b")
plt.fill_between(parameters_dict['alpha'], meanTestScore - stdTestScore, meanTestScore + stdTestScore, alpha=0.2, color="g")
plt.plot(parameters_dict['alpha'], meanTestScore, label="Cross-validation score (alpha)",
             color="r")

plt.legend(loc="best")
plt.xticks(parameters_dict['alpha'])
plt.show()

train_scores, test_scores = validation_curve(
    naive_bayes.MultinomialNB(fit_prior=False),transformedData, target, param_name="alpha", cv=5, 
    param_range=parameters_dict['alpha'],scoring="accuracy")

#Calculating mean and standard deviations of the scores
meanTrainScore = np.mean(train_scores, axis =1)
stdDevTrain = np.std(train_scores, axis=1)
meanTestScore = np.mean(test_scores, axis=1)
stdTestScore = np.std(test_scores, axis=1)

#Plotting 
plt.figure(figsize=(22,9))
#plt.subplot(2,2,1)
plt.title("Test-Train Accuracy Varying by alpha Without Learning Prior Probabilities")
plt.xlabel("alpha")
plt.ylabel("Accuracy")
plt.ylim(0.1, 1.05)
plt.fill_between(parameters_dict['alpha'], meanTrainScore - stdDevTrain, meanTrainScore + stdDevTrain, alpha=0.2, color="r")
plt.plot(parameters_dict['alpha'], meanTrainScore, label="Training score (alpha)",
             color="b")
plt.fill_between(parameters_dict['alpha'], meanTestScore - stdTestScore, meanTestScore + stdTestScore, alpha=0.2, color="g")
plt.plot(parameters_dict['alpha'], meanTestScore, label="Cross-validation score (alpha)",
             color="r")

plt.legend(loc="best")
plt.xticks(parameters_dict['alpha'])
plt.show()
```


![png](output_137_0.png)



![png](output_137_1.png)

We can see that the model with fit_priors = True has a stable and better accuracy than the model with fit_priors = False.

#### Nested Cross Validation With Fit Priors = True


```python
nBaiyes = GridSearchCV(naive_bayes.MultinomialNB(), parameters_dict, cv=inner_cv, scoring="accuracy", refit=True)
nested_score = cross_validate(nBaiyes, X=transformedData, y=target.ravel(), cv=outer_cv, scoring = make_scorer(scoringFunction))
print("Mean Accuracy : {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)
```

    Mean Accuracy : 0.86, Std Deviation: 0.03

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Std. Precision</th>
      <th>Mean Recall</th>
      <th>Std. Recall</th>
      <th>Mean F1</th>
      <th>Std F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.677829</td>
      <td>0.060970</td>
      <td>0.718626</td>
      <td>0.072895</td>
      <td>0.697229</td>
      <td>0.064510</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>0.511111</td>
      <td>0.163488</td>
      <td>0.258242</td>
      <td>0.105001</td>
      <td>0.333674</td>
      <td>0.116029</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>0.923815</td>
      <td>0.019058</td>
      <td>0.959504</td>
      <td>0.012369</td>
      <td>0.941288</td>
      <td>0.015249</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>0.937778</td>
      <td>0.081225</td>
      <td>0.415385</td>
      <td>0.104344</td>
      <td>0.567677</td>
      <td>0.090662</td>
    </tr>
  </tbody>
</table>
</div>

#### Nested Cross Validation With Fit Priors = False


```python
precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []
naiveBaiyesFalse = GridSearchCV(naive_bayes.MultinomialNB(fit_prior = False), parameters_dict, cv=inner_cv, scoring="accuracy", refit=True)
nested_score = cross_validate(naiveBaiyesFalse, X=transformedData, y=target.ravel(), cv=outer_cv, scoring = make_scorer(scoringFunction))
print("Mean Accuracy : {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)
```

    Mean Accuracy : 0.81, Std Deviation: 0.01

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Standard Deviation Precision</th>
      <th>Mean Recall</th>
      <th>Standard Deviation Recall</th>
      <th>Mean F1</th>
      <th>Standard Deviation F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.566968</td>
      <td>0.021852</td>
      <td>0.794463</td>
      <td>0.047642</td>
      <td>0.661349</td>
      <td>0.027873</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>0.488474</td>
      <td>0.085473</td>
      <td>0.900000</td>
      <td>0.057143</td>
      <td>0.630806</td>
      <td>0.081333</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.800826</td>
      <td>0.013171</td>
      <td>0.889339</td>
      <td>0.008113</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>0.698662</td>
      <td>0.075750</td>
      <td>0.969231</td>
      <td>0.061538</td>
      <td>0.809827</td>
      <td>0.061840</td>
    </tr>
  </tbody>
</table>
</div>

I have used Naive Baiyes only for the categorical dataset, because its not directly applicable to continuous data, unless we discretize numerical data.
I have measured the performance of Naive Baiyes in two hyperparameter settings one in which the model learns the set priors and one which does not.
The accuracy of the model that does not learn the priors has an accuracy of 81%, while the model that learns the priors has a mean accuracy of 86%, with a slightly higher standard deviation.
Both models are able to predict the Unacceptable the best followed by Very Good,Acceptable and Good. Both the models are not able to predict the Good class very well.

### Support Vector Machines

#### Creating parameter dictionary


```python
parameters_dict={"gamma":[0.01, 1.0,10,100]}
precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []
```

### Creating visualizations as to how different parameters affect the Accuracy of a SVM


```python
def createSVMViz(transformedData, target):
    train_scores, test_scores = validation_curve(
        svm.SVC(kernel="linear"),transformedData, target, param_name="gamma", cv=5, 
        param_range=parameters_dict['gamma'],scoring="accuracy")

    #Calculating mean and standard deviations of the scores
    meanTrainScore = np.mean(train_scores, axis =1)
    stdDevTrain = np.std(train_scores, axis=1)
    meanTestScore = np.mean(test_scores, axis=1)
    stdTestScore = np.std(test_scores, axis=1)

    #Plotting 
    plt.figure(figsize=(22,9))
    #plt.subplot(2,2,1)
    plt.title("Test-Train Accuracy Varying by gamma for Linear Kernel")
    plt.xlabel("gamma")
    plt.ylabel("Accuracy")
    plt.ylim(0.6, 1.05)
    plt.fill_between(parameters_dict['gamma'], meanTrainScore - stdDevTrain, meanTrainScore + stdDevTrain, alpha=0.2, color="r")
    plt.plot(parameters_dict['gamma'], meanTrainScore, label="Training score (gamma)",
                 color="b")
    plt.fill_between(parameters_dict['gamma'], meanTestScore - stdTestScore, meanTestScore + stdTestScore, alpha=0.2, color="g")
    plt.plot(parameters_dict['gamma'], meanTestScore, label="Cross-validation score (gamma)",
                 color="r")

    plt.legend(loc="best")
    plt.xticks(parameters_dict['gamma'])
    plt.show()

    #########################################################################################################

    train_scores, test_scores = validation_curve(
        svm.SVC(kernel="rbf"),transformedData, target, param_name="gamma", cv=5, 
        param_range=parameters_dict['gamma'],scoring="accuracy")

    #Calculating mean and standard deviations of the scores
    meanTrainScore = np.mean(train_scores, axis =1)
    stdDevTrain = np.std(train_scores, axis=1)
    meanTestScore = np.mean(test_scores, axis=1)
    stdTestScore = np.std(test_scores, axis=1)

    #Plotting 
    plt.figure(figsize=(22,9))
    #plt.subplot(2,2,1)
    plt.title("Test-Train Accuracy Varying by gamma for Radial Basis Function Kernel")
    plt.xlabel("gamma")
    plt.ylabel("Accuracy")
    plt.ylim(0.6, 1.05)
    plt.fill_between(parameters_dict['gamma'], meanTrainScore - stdDevTrain, meanTrainScore + stdDevTrain, alpha=0.2, color="r")
    plt.plot(parameters_dict['gamma'], meanTrainScore, label="Training score (gamma)",
                 color="b")
    plt.fill_between(parameters_dict['gamma'], meanTestScore - stdTestScore, meanTestScore + stdTestScore, alpha=0.2, color="g")
    plt.plot(parameters_dict['gamma'], meanTestScore, label="Cross-validation score (gamma)",
                 color="r")

    plt.legend(loc="best")
    plt.xticks(parameters_dict['gamma'])
    plt.show()

    #########################################################################################################

    train_scores, test_scores = validation_curve(
        svm.SVC(kernel="poly"),transformedData, target, param_name="gamma", cv=5, 
        param_range=parameters_dict['gamma'],scoring="accuracy")

    #Calculating mean and standard deviations of the scores
    meanTrainScore = np.mean(train_scores, axis =1)
    stdDevTrain = np.std(train_scores, axis=1)
    meanTestScore = np.mean(test_scores, axis=1)
    stdTestScore = np.std(test_scores, axis=1)

    #Plotting 
    plt.figure(figsize=(22,9))
    #plt.subplot(2,2,1)
    plt.title("Test-Train Accuracy Varying by gamma for Polynomial Kernel")
    plt.xlabel("gamma")
    plt.ylabel("Accuracy")
    plt.ylim(0.6, 1.05)
    plt.fill_between(parameters_dict['gamma'], meanTrainScore - stdDevTrain, meanTrainScore + stdDevTrain, alpha=0.2, color="r")
    plt.plot(parameters_dict['gamma'], meanTrainScore, label="Training score (gamma)",
                 color="b")
    plt.fill_between(parameters_dict['gamma'], meanTestScore - stdTestScore, meanTestScore + stdTestScore, alpha=0.2, color="g")
    plt.plot(parameters_dict['gamma'], meanTestScore, label="Cross-validation score (gamma)",
                 color="r")

    plt.legend(loc="best")
    plt.xticks(parameters_dict['gamma'])
    plt.show()
createSVMViz(transformedData, target)
```


![png](output_147_0.png)



![png](output_147_1.png)



![png](output_147_2.png)


### Support Vector Machine Continuous Data

#### Creating visualizations as to how different parameters affect the Accuracy of a SVM


```python
createSVMViz(transformedContData, target)
```


![png](output_150_0.png)



![png](output_150_1.png)

Models trained on both the types of data seem to perform almost equally on terms of stability but the models trained on the categorical data have better accuracy,.

#### Creating parameter dictionary


```python
parameters_dict={"gamma":[0.01, 1.0,10,100]}
precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []
```

### SVM with Linear Kernel


```python
svmLinear = GridSearchCV(svm.SVC(kernel="linear"), parameters_dict, cv=inner_cv, scoring="accuracy", refit=True)
nested_score = cross_validate(svmLinear, X=transformedData, y=target.ravel(), cv=outer_cv, scoring = make_scorer(scoringFunction))
print("Mean Accuracy : {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)
```

    Mean Accuracy : 0.93, Std Deviation: 0.02

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Standard Deviation Precision</th>
      <th>Mean Recall</th>
      <th>Standard Deviation Recall</th>
      <th>Mean F1</th>
      <th>Standard Deviation F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.853716</td>
      <td>0.026832</td>
      <td>0.882741</td>
      <td>0.033159</td>
      <td>0.867950</td>
      <td>0.029499</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>0.785077</td>
      <td>0.134238</td>
      <td>0.812088</td>
      <td>0.115327</td>
      <td>0.783245</td>
      <td>0.064822</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>0.973188</td>
      <td>0.009755</td>
      <td>0.959504</td>
      <td>0.011210</td>
      <td>0.966283</td>
      <td>0.009832</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>0.930476</td>
      <td>0.042228</td>
      <td>0.907692</td>
      <td>0.184615</td>
      <td>0.903492</td>
      <td>0.102614</td>
    </tr>
  </tbody>
</table>
</div>



### SVM with Linear Kernel Continuous Data


```python
svmLinear = GridSearchCV(svm.SVC(kernel="linear"), parameters_dict, cv=inner_cv, scoring="accuracy", refit=True)
nested_score = cross_validate(svmLinear, X=transformedContData, y=target.ravel(), cv=outer_cv, scoring = make_scorer(scoringFunction))
print("Mean Accuracy : {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)
```

    Mean Accuracy : 0.87, Std Deviation: 0.02

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Standard Deviation Precision</th>
      <th>Mean Recall</th>
      <th>Standard Deviation Recall</th>
      <th>Mean F1</th>
      <th>Standard Deviation F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.742725</td>
      <td>0.030356</td>
      <td>0.705502</td>
      <td>0.047256</td>
      <td>0.722967</td>
      <td>0.034844</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>0.760000</td>
      <td>0.068799</td>
      <td>0.768132</td>
      <td>0.094493</td>
      <td>0.756973</td>
      <td>0.035343</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>0.915086</td>
      <td>0.012252</td>
      <td>0.933058</td>
      <td>0.018918</td>
      <td>0.923842</td>
      <td>0.011176</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>0.914918</td>
      <td>0.078177</td>
      <td>0.815385</td>
      <td>0.092308</td>
      <td>0.860528</td>
      <td>0.077161</td>
    </tr>
  </tbody>
</table>
</div>

Linear Kernel seems to work better with categorical data than continuous data. The accuracy of the Linear Kernel Model trained on the categorical data is 93% while the other model has 87%. The first model predicts Unacceptable,Very Good, Acceptable and Good.
While the later model predicts Unacceptable, followed by Very Good, Good and Acceptable. 

### SVM with RBF Kernel


```python
precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []
svmLinearRBF = GridSearchCV(svm.SVC(kernel="rbf"), parameters_dict, cv=inner_cv, scoring="accuracy", refit=True)
nested_score = cross_validate(svmLinearRBF, X=transformedData, y=target.ravel(), cv=outer_cv, scoring = make_scorer(scoringFunction))
print("Mean Accuracy : {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)
```

    Mean Accuracy : 0.91, Std Deviation: 0.01

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Standard Deviation Precision</th>
      <th>Mean Recall</th>
      <th>Standard Deviation Recall</th>
      <th>Mean F1</th>
      <th>Standard Deviation F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.758702</td>
      <td>0.012814</td>
      <td>0.890738</td>
      <td>0.048159</td>
      <td>0.818649</td>
      <td>0.019893</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>0.200000</td>
      <td>0.400000</td>
      <td>0.028571</td>
      <td>0.057143</td>
      <td>0.050000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>0.964227</td>
      <td>0.013315</td>
      <td>0.995868</td>
      <td>0.003696</td>
      <td>0.979721</td>
      <td>0.005543</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.384615</td>
      <td>0.128717</td>
      <td>0.541871</td>
      <td>0.147759</td>
    </tr>
  </tbody>
</table>
</div>



### SVM with RBF Kernel Continuous Data


```python
precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []
svmLinearRBF = GridSearchCV(svm.SVC(kernel="rbf"), parameters_dict, cv=inner_cv, scoring="accuracy", refit=True)
nested_score = cross_validate(svmLinearRBF, X=transformedContData, y=target.ravel(), cv=outer_cv, scoring = make_scorer(scoringFunction))
print("Mean Accuracy : {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)
```

    Mean Accuracy : 0.99, Std Deviation: 0.01

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Standard Deviation Precision</th>
      <th>Mean Recall</th>
      <th>Standard Deviation Recall</th>
      <th>Mean F1</th>
      <th>Standard Deviation F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.966395</td>
      <td>0.010733</td>
      <td>0.976589</td>
      <td>0.022328</td>
      <td>0.971411</td>
      <td>0.015885</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>0.955714</td>
      <td>0.036564</td>
      <td>0.928571</td>
      <td>0.090351</td>
      <td>0.939542</td>
      <td>0.052670</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>0.994282</td>
      <td>0.006086</td>
      <td>0.999174</td>
      <td>0.001653</td>
      <td>0.996711</td>
      <td>0.003070</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>0.985714</td>
      <td>0.028571</td>
      <td>0.861538</td>
      <td>0.089707</td>
      <td>0.915752</td>
      <td>0.041119</td>
    </tr>
  </tbody>
</table>
</div>



SVM with RBF Kernel with continuous data data has a much higher accuracy than the model trained on categorical data. But the model trained on the categorical dataset is not able predict the Good class.

### SVM with Polynomial Kernel Categorical Dataset


```python
precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []
svmLinearPoly = GridSearchCV(svm.SVC(kernel="poly"), parameters_dict, cv=inner_cv, scoring="accuracy", refit=True)
nested_score = cross_validate(svmLinearPoly, X=transformedData, y=target.ravel(), cv=outer_cv, scoring = make_scorer(scoringFunction))
print("Mean Accuracy : {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)
```

    Mean Accuracy : 1.00, Std Deviation: 0.00

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Standard Deviation Precision</th>
      <th>Mean Recall</th>
      <th>Standard Deviation Recall</th>
      <th>Mean F1</th>
      <th>Standard Deviation F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.997436</td>
      <td>0.005128</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.998710</td>
      <td>0.002581</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.999174</td>
      <td>0.001653</td>
      <td>0.999586</td>
      <td>0.000828</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>

This is the best performning model. SVM with a polynomial kernel has the best predictive performance and almost has perfect precision and recall for all classes with an accuracy of 1.

### KNN Using Categorical Data


```python
# Create range of values for parameter
def createKNNViz(transformedData, target):
    neighbor = range(1,25,1)

    train_scores, test_scores = validation_curve(
        neighbors.KNeighborsClassifier(), transformedData, target, param_name="n_neighbors", cv=10, 
        param_range=neighbor,
        scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(22,9))
    plt.title("Validation Curve with K-NN Neighbors")
    plt.xlabel("k neighbors")
    plt.ylabel("accuracy")
    plt.ylim(0.6, 1.1)

    # Plot the values for recall with K ranging from 1-20
    plt.plot(neighbor, train_scores_mean, label="Training score",
                 color="r")
    plt.fill_between(neighbor, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="b")

    plt.plot(neighbor, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(neighbor, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="r")
    plt.legend(loc="best")
    plt.xticks(neighbor)

    plt.show()
createKNNViz(transformedData, target)
```


![png](output_167_0.png)


### KNN Visualization on Continuous Dataset


```python
createKNNViz(transformedContData, target)
```


![png](output_169_0.png)


We can see from the graph that KNN significantly performs better on the continuous dataset than on the categorical dataset. The accuracy is lower and the standard deviation of the accuracy is also higher on the dataset which has one-hot encoding.

#### Model Performance on Categorical Dataset


```python
KNNParamsTuned = {'n_neighbors':range(1,25,1)}

precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []

clfTree = GridSearchCV(neighbors.KNeighborsClassifier(), KNNParamsTuned, cv=inner_cv, scoring="accuracy", refit=True)
nested_score = cross_validate(clfTree, X=transformedData, y=target, cv=outer_cv, scoring = make_scorer(scoringFunction))
print("Mean Accuracy : {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)
```

    Mean Accuracy : 0.91, Std Deviation: 0.01

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Standard Deviation Precision</th>
      <th>Mean Recall</th>
      <th>Standard Deviation Recall</th>
      <th>Mean F1</th>
      <th>Standard Deviation F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.799840</td>
      <td>0.031963</td>
      <td>0.843746</td>
      <td>0.031822</td>
      <td>0.820378</td>
      <td>0.018810</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>0.719048</td>
      <td>0.151635</td>
      <td>0.318681</td>
      <td>0.095989</td>
      <td>0.438730</td>
      <td>0.121237</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>0.953012</td>
      <td>0.008936</td>
      <td>0.985950</td>
      <td>0.008509</td>
      <td>0.969135</td>
      <td>0.003452</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>0.980000</td>
      <td>0.040000</td>
      <td>0.600000</td>
      <td>0.089707</td>
      <td>0.738855</td>
      <td>0.065964</td>
    </tr>
  </tbody>
</table>
</div>



#### Model performance on Continuous Dataset


```python
KNNParamsTuned = {'n_neighbors':range(1,25,1)}

precisionArrayAcc = []
recallArrayAcc = []
f1ArrayAcc = []

precisionArrayGood = []
recallArrayGood = []
f1ArrayGood = []

precisionArrayUnacc = []
recallArrayUnacc = []
f1ArrayUnacc = []

precisionArrayVgood = []
recallArrayVgood = []
f1ArrayVgood = []

clfTree = GridSearchCV(neighbors.KNeighborsClassifier(), KNNParamsTuned, cv=inner_cv, scoring="accuracy", refit=True)
nested_score = cross_validate(clfTree, X=transformedContData, y=target, cv=outer_cv, scoring = make_scorer(scoringFunction))
print("Mean Accuracy : {0:.2f}, Std Deviation: {1:.2f}".format(nested_score['test_score'].mean(),nested_score['test_score'].std()))
createScoreDataFrame(precisionArrayAcc,precisionArrayGood,precisionArrayUnacc,precisionArrayVgood,\
                        recallArrayAcc,recallArrayGood,recallArrayUnacc,recallArrayVgood,\
                        f1ArrayAcc,f1ArrayGood,f1ArrayUnacc,f1ArrayVgood)
```

    Mean Accuracy : 0.95, Std Deviation: 0.01

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean Precision</th>
      <th>Standard Deviation Precision</th>
      <th>Mean Recall</th>
      <th>Standard Deviation Recall</th>
      <th>Mean F1</th>
      <th>Standard Deviation F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acceptable</th>
      <td>0.885215</td>
      <td>0.021711</td>
      <td>0.919310</td>
      <td>0.009447</td>
      <td>0.901798</td>
      <td>0.012711</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>0.851109</td>
      <td>0.077155</td>
      <td>0.710989</td>
      <td>0.087953</td>
      <td>0.769926</td>
      <td>0.060482</td>
    </tr>
    <tr>
      <th>Unacceptable</th>
      <td>0.977126</td>
      <td>0.001980</td>
      <td>0.988430</td>
      <td>0.003092</td>
      <td>0.982743</td>
      <td>0.002107</td>
    </tr>
    <tr>
      <th>Very Good</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.723077</td>
      <td>0.142671</td>
      <td>0.830532</td>
      <td>0.105921</td>
    </tr>
  </tbody>
</table>
</div>



KNN on the continuous dataset is able has a perfect precision for the "Very Good" class. But the performance is best for Unacceptable, followed by Acceptable, Very Good and Good.

SVM with Polynomial Kernel trained on the Categorical Dataset has the best predictive performance out of all the models.


```python

```
