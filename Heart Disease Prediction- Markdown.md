<div class="cell markdown">

### Logistic Regression

</div>

<div class="cell markdown">

Logistic Regression was used in the biological sciences in early
twentieth century.<br> It was then used in many social science
applications.<br> Logistic Regression is used when the dependent
variable(target) is categorical.<br> For example,<br> To predict whether
an email is spam (1) or (0)<br> Whether the tumor is malignant (1) or
not (0)<br>

</div>

<div class="cell markdown">

Types of Logistic Regression<br><br> <b>1. Binary Logistic
Regression</b><br> The categorical response has only two 2 possible
outcomes. Example: Spam or Not<br><br> <b>2. Multinomial Logistic
Regression</b><br> Three or more categories without ordering. Example:
Predicting which food is preferred more (Veg, Non-Veg, Vegan)<br><br>
<b>3. Ordinal Logistic Regression</b><br> Three or more categories with
ordering. Example: Movie rating from 1 to 5<br><br>

</div>

<div class="cell markdown">

In this notebook we are talking only about <b>Binary Logistic
Regression.</b>

</div>

<div class="cell code" data-execution_count="1">

``` python
%load_ext nb_black
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="2">

``` python
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="3">

``` python
# %matplotlib notebook
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

### Heart Disease Dataset :<br>

<b>Attribute Information:</b> <ol> <li>age</li> <li>sex</li> <li>chest
pain type (4 values)</li> <li>resting blood pressure</li> <li>serum
cholestoral in mg/dl</li> <li>fasting blood sugar \> 120 mg/dl</li>
<li>resting electrocardiographic results (values 0,1,2),</li>
<li>maximum heart rate achieved</li> <li>exercise induced angina</li>
<li>oldpeak = ST depression induced by exercise relative to rest</li>
<li>the slope of the peak exercise ST segment</li> <li>number of major
vessels (0-3) colored by flourosopy</li> <li>thal: 3 = normal; 6 = fixed
defect; 7 = reversable defect</li> </ol>

</div>

<div class="cell markdown">

<a href='https://www.kaggle.com/ronitf/heart-disease-uci'>Heart Disease
Dataset Link </a>

</div>

<div class="cell markdown">

![title](1.jpg)

</div>

<div class="cell markdown">

Read Full Paper
<a href = 'https://www.ijrte.org/wp-content/uploads/papers/v8i2S3/B11630782S319.pdf'>Here</a>

</div>

<div class="cell markdown">

#### Why you want to apply classification on selected dataset? Discuss full story behind dataset.

In this dataset we have some parameters like age , sex , chest pain type
and etc. and finally we have a target which is tell us weather a person
has a heart disease or not.From all this parameters we have to predict
weather person has a heart disease or not i.e we have to predict 0 ( No
heart Disease) or 1 ( Yes heart Disease ). So it is a classification
problem we have to classify the categoris and that's why we apply
classification algorithms on this dataset.<br><br> Generally for two
categories we called it <b>Binary classification.</b><br> For more than
two categories we called it <b>Multiclass classification</b><br>

</div>

<div class="cell code" data-execution_count="6">

``` python
dataset = pd.read_csv("heart.csv")
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="7">

``` python
dataset.head()
```

<div class="output execute_result" data-execution_count="7">

``` 
   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \
0   63    1   3       145   233    1        0      150      0      2.3      0   
1   37    1   2       130   250    0        1      187      0      3.5      0   
2   41    0   1       130   204    0        0      172      0      1.4      2   
3   56    1   1       120   236    0        1      178      0      0.8      2   
4   57    0   0       120   354    0        1      163      1      0.6      2   

   ca  thal  target  
0   0     1       1  
1   0     2       1  
2   0     2       1  
3   0     2       1  
4   0     2       1  
```

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="8">

``` python
dataset.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 303 entries, 0 to 302
    Data columns (total 14 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       303 non-null    int64  
     1   sex       303 non-null    int64  
     2   cp        303 non-null    int64  
     3   trestbps  303 non-null    int64  
     4   chol      303 non-null    int64  
     5   fbs       303 non-null    int64  
     6   restecg   303 non-null    int64  
     7   thalach   303 non-null    int64  
     8   exang     303 non-null    int64  
     9   oldpeak   303 non-null    float64
     10  slope     303 non-null    int64  
     11  ca        303 non-null    int64  
     12  thal      303 non-null    int64  
     13  target    303 non-null    int64  
    dtypes: float64(1), int64(13)
    memory usage: 33.3 KB

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

#### How many total observations in data?

There are total 303 non-null values in dataset.

</div>

<div class="cell markdown">

#### How many independent variables?

There are 12 independent variable in this dataset.<br> <ul> <li>age</li>
<li>sex</li> <li>cp</li> <li>trestbps</li> <li>chol</li> <li>fbs </li>
<li>restecg</li> <li>thalach</li> <li>exang </li> <li>oldpeak</li>
<li>slope</li> <li>ca</li> <li>thal</li> </ul>

</div>

<div class="cell markdown">

#### Which is dependent variable?

' target ' is dependent variable and that we have to predict.

</div>

<div class="cell markdown">

#### Heart Disease Dataset :<br>

<b>Attribute Information:</b> <ol> <li>age</li> <li>sex</li> <li>chest
pain type (4 values)</li> <li>resting blood pressure</li> <li>serum
cholestoral in mg/dl</li> <li>fasting blood sugar \> 120 mg/dl</li>
<li>resting electrocardiographic results (values 0,1,2),</li>
<li>maximum heart rate achieved</li> <li>exercise induced angina</li>
<li>oldpeak = ST depression induced by exercise relative to rest</li>
<li>the slope of the peak exercise ST segment</li> <li>number of major
vessels (0-3) colored by flourosopy</li> <li>thal: 3 = normal; 6 = fixed
defect; 7 = reversable defect</li> </ol>

</div>

<div class="cell code" data-execution_count="4">

``` python
import seaborn as sns
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="186">

``` python
sns.displot(dataset["thalach"], kind="kde")
plt.xlabel("maximum heart rate achieved")
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="186">

    Text(0.5, 9.444444444444438, 'maximum heart rate achieved')

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="9">

``` python
dataset["thal"].value_counts()
```

<div class="output execute_result" data-execution_count="9">

    2    166
    3    117
    1     18
    0      2
    Name: thal, dtype: int64

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

##### Categorical Values

sex<br> cp<br> exang<br> restecg<br> slope<br> ca<br> thal<br>

</div>

<div class="cell code" data-execution_count="10">

``` python
dataset.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 303 entries, 0 to 302
    Data columns (total 14 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       303 non-null    int64  
     1   sex       303 non-null    int64  
     2   cp        303 non-null    int64  
     3   trestbps  303 non-null    int64  
     4   chol      303 non-null    int64  
     5   fbs       303 non-null    int64  
     6   restecg   303 non-null    int64  
     7   thalach   303 non-null    int64  
     8   exang     303 non-null    int64  
     9   oldpeak   303 non-null    float64
     10  slope     303 non-null    int64  
     11  ca        303 non-null    int64  
     12  thal      303 non-null    int64  
     13  target    303 non-null    int64  
    dtypes: float64(1), int64(13)
    memory usage: 33.3 KB

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="11">

``` python
dataset.head(10)
```

<div class="output execute_result" data-execution_count="11">

``` 
   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \
0   63    1   3       145   233    1        0      150      0      2.3      0   
1   37    1   2       130   250    0        1      187      0      3.5      0   
2   41    0   1       130   204    0        0      172      0      1.4      2   
3   56    1   1       120   236    0        1      178      0      0.8      2   
4   57    0   0       120   354    0        1      163      1      0.6      2   
5   57    1   0       140   192    0        1      148      0      0.4      1   
6   56    0   1       140   294    0        0      153      0      1.3      1   
7   44    1   1       120   263    0        1      173      0      0.0      2   
8   52    1   2       172   199    1        1      162      0      0.5      2   
9   57    1   2       150   168    0        1      174      0      1.6      2   

   ca  thal  target  
0   0     1       1  
1   0     2       1  
2   0     2       1  
3   0     2       1  
4   0     2       1  
5   0     1       1  
6   0     2       1  
7   0     3       1  
8   0     3       1  
9   0     2       1  
```

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="12">

``` python
categorized_sex = pd.get_dummies(dataset["sex"])  # caterogies encoder
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="13">

``` python
categorized_sex.columns = ["female", "male"]
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="14">

``` python
categorized_data = pd.get_dummies(
    data=dataset,
    columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"],
    drop_first=True,
)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

### Model prepration

for this time we take all the variables and check accuracy. And We make
this model as the base model

</div>

<div class="cell code" data-execution_count="15">

``` python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="16">

``` python
sc = StandardScaler()
targets = categorized_data["target_1"]
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="17">

``` python
X = categorized_data.drop(columns=["target_1"])
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="18">

``` python
new_X = sc.fit_transform(X)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="19">

``` python
X_train, X_test, y_train, y_test = train_test_split(
    new_X, targets, test_size=0.2, random_state=0
)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="20">

``` python
model = LogisticRegression()
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="21">

``` python
model.fit(X_train, y_train)
```

<div class="output execute_result" data-execution_count="21">

    LogisticRegression()

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="22">

``` python
test_predictions = model.predict(X_test)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="23">

``` python
print(classification_report(y_test, test_predictions))
```

<div class="output stream stdout">

``` 
              precision    recall  f1-score   support

           0       0.88      0.85      0.87        27
           1       0.89      0.91      0.90        34

    accuracy                           0.89        61
   macro avg       0.89      0.88      0.88        61
weighted avg       0.89      0.89      0.88        61

```

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="24">

``` python
print(classification_report(y_train, model.predict(X_train)))
```

<div class="output stream stdout">

``` 
              precision    recall  f1-score   support

           0       0.88      0.82      0.85       111
           1       0.86      0.91      0.88       131

    accuracy                           0.87       242
   macro avg       0.87      0.86      0.87       242
weighted avg       0.87      0.87      0.87       242

```

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="25">

``` python
model.score(X_test, y_test)
```

<div class="output execute_result" data-execution_count="25">

    0.8852459016393442

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="26">

``` python
model.score(X_train, y_train)
```

<div class="output execute_result" data-execution_count="26">

    0.8677685950413223

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Test Accuracy : 88% <br> <br> Train Accuracy : 86% <br></b><br>

</div>

<div class="cell markdown">

### Corelation

</div>

<div class="cell code" data-execution_count="27">

``` python
dataset.corr()["target"]
```

<div class="output execute_result" data-execution_count="27">

    age        -0.225439
    sex        -0.280937
    cp          0.433798
    trestbps   -0.144931
    chol       -0.085239
    fbs        -0.028046
    restecg     0.137230
    thalach     0.421741
    exang      -0.436757
    oldpeak    -0.430696
    slope       0.345877
    ca         -0.391724
    thal       -0.344029
    target      1.000000
    Name: target, dtype: float64

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

We take cp , restecg , thalach , slope variables in count

</div>

<div class="cell code" data-execution_count="28">

``` python
categorized_data.corr()["target_1"]
```

<div class="output execute_result" data-execution_count="28">

    age         -0.225439
    trestbps    -0.144931
    chol        -0.085239
    thalach      0.421741
    oldpeak     -0.430696
    sex_1       -0.280937
    cp_1         0.245879
    cp_2         0.316742
    cp_3         0.086957
    fbs_1       -0.028046
    restecg_1    0.175322
    restecg_2   -0.068410
    exang_1     -0.436757
    slope_1     -0.362053
    slope_2      0.394066
    ca_1        -0.232412
    ca_2        -0.273998
    ca_3        -0.210615
    ca_4         0.066441
    thal_1      -0.106589
    thal_2       0.527334
    thal_3      -0.486112
    target_1     1.000000
    Name: target_1, dtype: float64

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

### Modal Improvement I

First time we take all the variables and check how our model
performs.<br> As you can see above most of the variables are negative
corelated and that thing is bad for model.<br> So for this time we
include most of the positive corelated variables and check on that data
how our model performes.<br>

</div>

<div class="cell markdown">

##### Highly Corelated Variables

thalach cp restecg slope thal

</div>

<div class="cell code" data-execution_count="29">

``` python
X = categorized_data.drop(columns=["target_1"])
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="30">

``` python
X = X[["thalach", "cp_1", "cp_2", "cp_3", "restecg_1", "slope_2", "thal_2"]]
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="31">

``` python
new_x = sc.fit_transform(X)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="32">

``` python
X_train, X_test, y_train, y_test = train_test_split(
    new_x, targets, test_size=0.2, random_state=2
)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="33">

``` python
model2 = LogisticRegression()
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="34">

``` python
model2.fit(X_train, y_train)
```

<div class="output execute_result" data-execution_count="34">

    LogisticRegression()

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="35">

``` python
model2.score(X_test, y_test)
```

<div class="output execute_result" data-execution_count="35">

    0.8688524590163934

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="36">

``` python
model2.score(X_train, y_train)
```

<div class="output execute_result" data-execution_count="36">

    0.7768595041322314

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Test Accuracy : 86% <br> <br> Train Accuracy : 77% <br></b><br>

</div>

<div class="cell code" data-execution_count="37">

``` python
test_predictions = model2.predict(X_test)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="38">

``` python
print(classification_report(y_test, test_predictions))
```

<div class="output stream stdout">

``` 
              precision    recall  f1-score   support

           0       0.93      0.81      0.87        32
           1       0.82      0.93      0.87        29

    accuracy                           0.87        61
   macro avg       0.87      0.87      0.87        61
weighted avg       0.88      0.87      0.87        61

```

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="39">

``` python
print(classification_report(y_train, model2.predict(X_train)))
```

<div class="output stream stdout">

``` 
              precision    recall  f1-score   support

           0       0.74      0.76      0.75       106
           1       0.81      0.79      0.80       136

    accuracy                           0.78       242
   macro avg       0.77      0.78      0.77       242
weighted avg       0.78      0.78      0.78       242

```

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

#### Now we can say our model1 is best..\!\!

</div>

<div class="cell markdown">

Becuase model1 has accuracy of 88% and model2 has accuracy of
86%<br><br> model1 has train accuracy is 87% which near to its test
accuracy on otherside model2 has train accuracy 77% which is very far
from its test accuracy.<br><br>That means our model1 can perform very
good on train data as well as test data.

</div>

<div class="cell markdown">

### Cross Validation

</div>

<div class="cell code" data-execution_count="40">

``` python
from sklearn.model_selection import cross_val_score
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="41">

``` python
scores_model1 = cross_val_score(model, new_X, targets, cv=6, n_jobs=-1)
scores_model2 = cross_val_score(model2, new_x, targets, cv=6, n_jobs=-1)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="42">

``` python
scores_model1.mean()
```

<div class="output execute_result" data-execution_count="42">

    0.8611764705882353

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="43">

``` python
scores_model2.mean()
```

<div class="output execute_result" data-execution_count="43">

    0.8016993464052287

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Now, We can prove that model1 is best by using K-Fold Cross
Validation.</b>

</div>

<div class="cell code" data-execution_count="44">

``` python
dataset.head(10)
```

<div class="output execute_result" data-execution_count="44">

``` 
   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \
0   63    1   3       145   233    1        0      150      0      2.3      0   
1   37    1   2       130   250    0        1      187      0      3.5      0   
2   41    0   1       130   204    0        0      172      0      1.4      2   
3   56    1   1       120   236    0        1      178      0      0.8      2   
4   57    0   0       120   354    0        1      163      1      0.6      2   
5   57    1   0       140   192    0        1      148      0      0.4      1   
6   56    0   1       140   294    0        0      153      0      1.3      1   
7   44    1   1       120   263    0        1      173      0      0.0      2   
8   52    1   2       172   199    1        1      162      0      0.5      2   
9   57    1   2       150   168    0        1      174      0      1.6      2   

   ca  thal  target  
0   0     1       1  
1   0     2       1  
2   0     2       1  
3   0     2       1  
4   0     2       1  
5   0     1       1  
6   0     2       1  
7   0     3       1  
8   0     3       1  
9   0     2       1  
```

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<ol>
    <li>age</li> 
    <li>sex</li>
    <li>chest pain type (4 values)</li>
    <li>resting blood pressure</li>
    <li>serum cholestoral in mg/dl</li>
    <li>fasting blood sugar > 120 mg/dl</li>
    <li>resting electrocardiographic results (values 0,1,2),</li>
    <li>maximum heart rate achieved</li>
    <li>exercise induced angina</li>
    <li>oldpeak = ST depression induced by exercise relative to rest</li>
    <li>the slope of the peak exercise ST segment</li>
    <li>number of major vessels (0-3) colored by flourosopy</li>
    <li>thal: 3 = normal; 6 = fixed defect; 7 = reversable defect</li>
</ol>

</div>

<div class="cell code" data-execution_count="106">

``` python
%matplotlib notebook
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

### Model Improvment II

now we do analysis on all the variables and after graphical
representation we select the variable and then make a model and then
check K-Fold Cross Validation Score.

</div>

<div class="cell code" data-execution_count="105">

``` python
sns.violinplot(x="target", y="thalach", data=dataset)  # 0 female and 1 male
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="105">

    <AxesSubplot:xlabel='target', ylabel='thalach'>

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Observation:</b><br> As you can see a person with high heart rate
have a little chance of heart disease. In the heart disease positive you
can see very high desntiy at mean value.A person with aproximate heart
rate of 160 have e very good chance of heart disease.That's why include
this field in model prepration.

</div>

<div class="cell code" data-execution_count="117">

``` python
sns.barplot(x="fbs", y="target", data=dataset)  # 0 female and 1 male
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="117">

    <AxesSubplot:xlabel='fbs', ylabel='target'>

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Observation:</b><br> For both cases there is not a large difference
fbs (fasting blood pressure). So this field is not that much userful. so
we are not including this field.

</div>

<div class="cell code" data-execution_count="118">

``` python
sns.countplot(x="sex", hue="target", data=dataset, palette="mako")
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="118">

    <AxesSubplot:xlabel='sex', ylabel='count'>

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Observation:</b><br> In this plot we can see women have low chance of
heart disease and men have high chance of high disease.But women with
heart disease bar also near to men with heart disease.so we are
including that field in model prepration

</div>

<div class="cell code" data-execution_count="119">

``` python
sns.countplot(x="thal", hue="target", data=dataset, palette="rocket")
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="119">

    <AxesSubplot:xlabel='thal', ylabel='count'>

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Observation:</b><br> As we can see people with thal type-2 have very
high chance of heart disease and people with thal type-3 have very good
chance not have a heart disease. So we use this field in the model
prepration.

</div>

<div class="cell code" data-execution_count="120">

``` python
sns.countplot(x="exang", hue="target", data=dataset, palette=["violet", "red"])
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="120">

    <AxesSubplot:xlabel='exang', ylabel='count'>

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Observation:</b><br> As we can see people who don't excersice have a
very good chance of heart disease and people who do excersice have ver
less chance of heart disease.

</div>

<div class="cell code" data-execution_count="121">

``` python
sns.countplot(x="ca", hue="target", data=dataset)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="121">

    <AxesSubplot:xlabel='ca', ylabel='count'>

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Observation:</b><br> people with ca type-0 have a very good chance of
having a heart disease and ca type-3 have less chance to have a heart
disease. So that why we include only ca type-0 in model prepration.

</div>

<div class="cell code" data-execution_count="122">

``` python
sns.countplot(x="restecg", hue="target", data=dataset, palette="coolwarm")
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="122">

    <AxesSubplot:xlabel='restecg', ylabel='count'>

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Observation:</b><br> People with ST-T (type - 1) as
electrocariographic result have a good chance to having a heart disease.
People with hypertrophy ( type-2 ) as electrocardiographic result have
not that much enough data. so that we can't predict.It has a good
corlation with target so that we include this field.

</div>

<div class="cell code" data-execution_count="125">

``` python
sns.violinplot(x="target", y="chol", hue="sex", data=dataset, palette="magma")
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="125">

    <AxesSubplot:xlabel='target', ylabel='chol'>

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Observation:</b><br> we can see for all the results we have near
equal mean. we can not classify the data by using cholestrol field. It
is also negative correlated with target so that's why we are not include
this field.

</div>

<div class="cell code" data-execution_count="124">

``` python
sns.countplot(x="fbs", hue="target", data=dataset, palette=["blue", "orange"])
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="124">

    <AxesSubplot:xlabel='fbs', ylabel='count'>

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Observation:</b><br> Surprisingly...\!\!\!<br> Person with blood
presure \< 120 have a high chance of high diseases. For person with high
blood presure we can't say anything. So it is unpredicteble we are not
include this field.

</div>

<div class="cell code" data-execution_count="128">

``` python
sns.barplot(x="cp", y="oldpeak", hue="target", data=dataset, palette="rocket")
# plt.legend()
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="128">

    <AxesSubplot:xlabel='cp', ylabel='oldpeak'>

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Observation:</b><br> person with chest pain type-0 and type-2 and
high oldpeak have a very good chance of having heart disease. for type-3
and type-2 we can say neutral. We include both field in model
prepration.

</div>

<div class="cell code" data-execution_count="132">

``` python
sns.countplot(x="ca", hue="target", data=dataset, palette="rocket")
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="132">

    <AxesSubplot:xlabel='ca', ylabel='count'>

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Observation:</b><br> We can see that ca type-0 have a very good
chance of having heart disease then other types. we include this field
and check how it performs on model.

</div>

<div class="cell code" data-execution_count="134">

``` python
sns.countplot(x="slope", hue="target", data=dataset, palette="coolwarm")
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output execute_result" data-execution_count="134">

    <AxesSubplot:xlabel='slope', ylabel='count'>

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Observation:</b><br> We can clearly see that person with slop type-2
(downsloping) have a very good chance of having heart diseas and person
with slope type-1 (flat) have less chance of having heart disease. we
use this field in model preparation.

</div>

<div class="cell markdown">

<b>Include variables</b><br> <ul> <li>thalach</li> <li>ca</li>
<li>oldpeak</li> <li>cp</li> <li>restecg</li> <li>ca</li> <li>exang</li>
<li>thal</li> <li>trestbps</li> <li>slope</li> <li>age</li> <li>sex</li>

</ul>

<b>Not Included variables</b><br> <ul> <li>fbs</li> <li>chol</li>  
</ul>

</div>

<div class="cell code" data-execution_count="45">

``` python
categorized_data.head()
```

<div class="output execute_result" data-execution_count="45">

``` 
   age  trestbps  chol  thalach  oldpeak  sex_1  cp_1  cp_2  cp_3  fbs_1  ...  \
0   63       145   233      150      2.3      1     0     0     1      1  ...   
1   37       130   250      187      3.5      1     0     1     0      0  ...   
2   41       130   204      172      1.4      0     1     0     0      0  ...   
3   56       120   236      178      0.8      1     1     0     0      0  ...   
4   57       120   354      163      0.6      0     0     0     0      0  ...   

   slope_1  slope_2  ca_1  ca_2  ca_3  ca_4  thal_1  thal_2  thal_3  target_1  
0        0        0     0     0     0     0       1       0       0         1  
1        0        0     0     0     0     0       0       1       0         1  
2        0        1     0     0     0     0       0       1       0         1  
3        0        1     0     0     0     0       0       1       0         1  
4        0        1     0     0     0     0       0       1       0         1  

[5 rows x 23 columns]
```

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="78">

``` python
X = categorized_data.drop(columns=["chol", "fbs_1", "target_1"])
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="79">

``` python
Y = categorized_data["target_1"]
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="80">

``` python
new_X1 = sc.fit_transform(X)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="81">

``` python
X_train, X_test, y_train, y_test = train_test_split(
    new_X1, Y, test_size=0.2, random_state=0
)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="82">

``` python
model3 = LogisticRegression()
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="83">

``` python
model3.fit(X_train, y_train)
# model3_b.fit(X_train, y_train)
```

<div class="output execute_result" data-execution_count="83">

    LogisticRegression()

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="84">

``` python
model3.score(X_test, y_test)
```

<div class="output execute_result" data-execution_count="84">

    0.8524590163934426

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="85">

``` python
model3.score(X_train, y_train)
```

<div class="output execute_result" data-execution_count="85">

    0.8760330578512396

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Test Accuracy : 85.24% <br> <br> Train Accuracy : 85.95% <br></b><br>

</div>

<div class="cell code" data-execution_count="86">

``` python
print(classification_report(y_test, model3.predict(X_test)))
```

<div class="output stream stdout">

``` 
              precision    recall  f1-score   support

           0       0.82      0.85      0.84        27
           1       0.88      0.85      0.87        34

    accuracy                           0.85        61
   macro avg       0.85      0.85      0.85        61
weighted avg       0.85      0.85      0.85        61

```

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="87">

``` python
print(classification_report(y_train, model3.predict(X_train)))
```

<div class="output stream stdout">

``` 
              precision    recall  f1-score   support

           0       0.89      0.84      0.86       111
           1       0.87      0.91      0.89       131

    accuracy                           0.88       242
   macro avg       0.88      0.87      0.87       242
weighted avg       0.88      0.88      0.88       242

```

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="88">

``` python
scores_model3 = cross_val_score(model3, new_X1, Y, cv=6, n_jobs=-1)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell code" data-execution_count="89">

``` python
scores_model1.mean(), scores_model2.mean(), scores_model3.mean()
```

<div class="output execute_result" data-execution_count="89">

    (0.8611764705882353, 0.8016993464052287, 0.8643137254901961)

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

</div>

<div class="cell markdown">

<b>Conclusion :</b><br> Model 3 performs slight better than Model 1.<br>
It gives 86.43 % accuracy by using Cross Validation while Model 1 Gives
86.11% accuracy.<br> Model 2 Perform worst in all three models because
it is not include negative corelated features and only include positive
corelated features. That's why Model2 not perform well.

</div>

<div class="cell code">

``` python
```

</div>
