# -Poor-Health-Likelihood 📝
## Data Source 📊 [NPHA](https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha)).

## Introduction 🟠
Three models (Neural Networks, Random Forest, and Linear Regression) are used to predict outcomes based on input data, and the models' performance is cross-validated using k-fold and LOOCV to enable comparison of each model's performance.
Since the focus of the analysis is to predict the likelihood of having poor health, The target variable (Physical Health) values, which include 1,2,3,4 and 5 representing Excellent, Very Good, Good, Fair, and Poor, respectively, transformed into a binary format suitable for binary classification, where the model predicts whether an individual's physical health score is 5 (coded as 1) or not 5 (coded as 0). This also means the primary form of logistic regression is used to model the binary variable and not multinominal logistic regression.
Neural Networks (NNs) are used to learn from the complex patterns in the dataset and predict various health outcomes.
Random Forests are used because they can handle overfitting better than individual decision trees and are suitable for dealing with a mix of numerical and categorical data, which describes the given dataset.
This Project attempts to use these three algorithms to predict the likelihood of Poor Health.

## Exploratory Data Analysis (EDA) 🟢
At this stage, the given dataset is Pre-processed and analyzed. It consists of different phases and may include Data preparation, Feature engineering, Data wrangling, Data reduction, and Feature engineering. 
Before EDA, essential functions are performed on the given data to understand and clean it. Info () function is run on the data frame. This provides valuable information, such as the Number of columns, column labels, column data types, and the Number of cells in each column. With this, insight can easily be captured into which features are the target and independent variables.

### Checking for Duplication: 🟢
When analyzing the data set, it was found that all columns are categorical with encoded values. Also, two types of categories are present in the dataset: Nominal and ordinal features. Nominal features represent categories without inherent order, while ordinal features have a specific order or ranking. In the data set, Nominal Features include Gender, Race, Employment, Stress Keeps Patient from Sleeping, Medication Keeps Patient from Sleeping, Pain Keeps Patient from Sleeping, Bathroom Needs Patient from Sleeping, Uknown Keeps Patient from Sleeping, and Prescription Sleep Medication. In contrast, ordinal features include the Number of doctors visited, Age, Physical health, Mental health, Dental Health, and Trouble Sleeping. 
     Unique () function is used to check for duplicates. This function execution presented the idea that all data point values are unique. All columns in the dataset are also 

### Missing Values 🟢
The initial data set presented with no missing values but had -1 and some undefined values. Variable re-coding was used to replace data points with -1, which meant refused, and six, which had no meaning to the analysis; therefore, these were considered NAN values. Also, with the "Trouble Sleeping Column," numerical value 3 meant 0 nights, converted to NAN values. 
      After re-coding the data points, NAN values introduced for the "Trouble Sleeping" Column were 361, which constituted the largest column with null values that accounted for 50% of its values. The percentage of missing values for columns physical health, Mental health, Dental health, and Prescription sleep medication is ~0.14%, ~1%, ~8%, and ~0.42%.

### Conversion of Data Types 🟢
Converted data types of known categories to category data type. This reduces memory usage since categorical data is stored more efficiently than object types and improves performance.

### Feature Renaming 🟢
The feature name "Physcal Health" is incorrectly spelled so it is changed to "Physical Health."

### Data Reduction 🟢
In the dataset, "Trouble Sleeping" is dropped since almost 50% of its values are null values. “Uknown keeps patient from sleeping is also dropped as it does not impact the analysis. All other null values in columns such as, Physical health, Mental health, Dental health, and Prescription sleep medication is removed.

## Cross-tabulations of Categorical Variables against 'Physical Health' 🟠
Bar charts is used to visually assess the frequency of physical health ratings across levels of categorical variables like 'Employment' and 'Race'.

#### Employment
Among those working full-time (coded as 1), very few rated their physical health as poor (5). Part-time workers (coded as 2) did not report poor physical health.

Retirees (coded as 3) showed a more varied distribution across health ratings, with some reporting poor physical health.

Those not working at this time (coded as 4) had a small number, but a relatively higher proportion reported poor physical health.
 

#### Race
The majority of the responses are from individuals identified as White Non-Hispanic (coded as 1), with a small number reporting poor physical health. 

Other racial groups (coded as 2, 3, 4, and 5) have a relatively small sample size, but the distribution follows a similar trend, with few reporting poor physical health.

The box plot is used to visualize some relationships. Specifically to compare the distribution of numerical variables across different levels of physical health.

![image](https://github.com/samdesigncodes/-Poor-Health-Likelihood/assets/118144590/85c33c95-9cda-470a-9933-ab66e6712ef9)


### Dental Health vs Physical Health
Similarly, the distribution of dental health ratings shows a trend where poorer physical health is associated with poorer dental health. However, there is a considerable overlap in the interquartile ranges, indicating that while there is a trend, there are also many individuals whose dental health does not correspond directly with their physical health.

### Employment Status vs Physical Health
The bar chart reveals that:
Individuals who are retired (coded as 3) make up the largest group in the dataset and include most of the individuals reporting 'Poor' physical health.

The group of individuals not working at this time (coded as 4) is smaller, but a higher proportion report 'Poor' physical health compared to other employment statuses.

Full-time (coded as 1) and part-time workers (coded as 2) have fewer instances of 'Poor' physical health, which could be due to various factors, including age and socioeconomic status.

These visuals affirm that while there are trends in the data, health outcomes are complex and influenced by a combination of factors.
 
![image](https://github.com/samdesigncodes/-Poor-Health-Likelihood/assets/118144590/a86558d1-d167-42ca-a2a0-5cc035194324)

## Logistic Regression:
It is a statistical method for binary classification problems, and it estimates probabilities using a logistic function, which is an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1. Thus, it's most useful for predicting the probability of a binary outcome.

P(Y=1)= 1/(1+e−(β0+β1X1+...+βnXn)) where P(Y=1) is the probability of the target variable being in class 1,
X1,..., Xn are the predictor variables,

### Appropriateness for the Prediction of the Target Variable
This model is well-suited for predicting the likelihood of a specific outcome, having poor physical health or not, since the outcome is binary. It can show the relationship between each predictor and the outcome as an odds ratio, which can be valuable in health-related research akin to this Project.
Instead of predicting the values of the response variable as 0 or 1, we predict the probability of the response variable being 0 or 1. We instead want to make sure the curve fitted makes the range of the response variable y ∈ [0,1] belong to 0 and 1 and the covariant X ∈ R. 


![image](https://github.com/samdesigncodes/-Poor-Health-Likelihood/assets/118144590/1731e9cb-a84e-4829-a175-af2460509e85)

No matter where we are on the x-axis, between minus plus infinity, only values between 0 and 1 result.


## Random Forest:
This is an ensemble learning method based on decision tree models. It works by constructing a multitude of decision trees at training time and outputting the class, which is the mode of the classes (classification) of the individual trees.
It is based on the aggregation of results from multiple decision trees, each of which may have been trained on a subset of the data and features. The final prediction is averaged over the other predictions of the individual trees

### Appropriateness for Predicting the Target Variable
Random forest can capture complex, nonlinear relationships between features and the target variable without requiring transformations. It's robust against overfitting due to the ensemble approach and is suitable for handling mixed data types (numerical and categorical). These characteristics make it a strong candidate for predicting physical health since predictors may have complex interactions.

## Neural Networks:
It is a set of algorithms modelled loosely after the human brain, designed to recognize patterns. They interpret sensory data through machine perception, labelling, or clustering raw inputs.

The fundamental building block of a neural network is the neuron or node, which receives inputs (Xi) and a bias (b) and produces an output (y) through an activation function (f)

y=f(∑i(Xi⋅wi)+b) where wi are the weights. Complex networks consist of multiple layers of these neurons, allowing them to learn nonlinear relationships. 

### Appropriateness for the Prediction of the Target Variable
Neural Networks are highly flexible and can model complex, nonlinear relationships that might be present in the given dataset, including interactions between various health indicators that could influence physical health status. Their ability to handle large volumes of data and learn feature representations makes them particularly powerful for this Project, especially since the relationships between variables are complex.
![image](https://github.com/samdesigncodes/-Poor-Health-Likelihood/assets/118144590/7d4514ca-8439-481d-8539-4117af23e34c)

# IV.	RESULTS AND DISCUSSION
After applying the individual algorithms to the dataset, below is the analysis and description of the evaluation metrics

Model	Precision (Class 1)	Recall (Class 1)	F1-Score	Accuracy	Precision (Weighted Avg)	Recall (Weighted Avg)	F1-Score (weighted Avg)
Logistic Regression	0.00	0.00	0.00	0.95	0.91	0.95	0.93
Random Forest	0.00	0.00	0.00	0.95	0.91	0.95	0.93
Neural Network	0.00	0.00	0.00	0.95	0.91	0.95	0.93

From the table above, it was found that the precision class is 0.00 for all models, suggesting that none of the models could correctly predict positive cases of poor physical health. 

Recall: Measures the proportion of actual positives that were correctly identified; again, the score is 0.00 for all models, indicating that all positive class instances were missed.\

#### F1-Score: ]
A weighted average of precision and recall. Since both are 0, the F1-score is also 0, reflecting poor performance on the positive class.

#### Accuracy: 
Shows a high score of 0.95 for all models, which at first glance suggests good performance. However, this metric is misleading due to the imbalanced nature of the dataset, where the majority class (class 0) dominates.

#### Weighted Averages: 
Precision, recall, and F1-score weighted averages are relatively high, indicating that the models perform well on the majority class but fail to capture the minority class (class 1).

The purpose of the analysis is to predict the physical health status, identifying instances of 'Poor Physical Health' within the dataset. The high accuracy across models masks their inability to identify minority cases, which is crucial for the analysis. 
The model's failure to identify any instances of poor physical health is a significant concern, indicating a lack of sensitivity. This is likely due to the imbalance in the dataset, where instances of 'Poor' physical health are far fewer than 'Not Poor.'

## Addressing the Imbalance
To address the imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) is used to oversample the minority class or undersample the majority class to balance the dataset. In this case, SMOTE is used to generate synthetic examples for the minority class by interpolating existing examples. This helped to balance the class distribution and potentially improve the model performance on the minority class.

### SMOTE Application
After applying SMOTE to balance the dataset and retraining the individual models, below is an analysis of the evaluation metrics for Logistic Regression, Rain Forest, and Neural Network Models. 

Model	Precision (Class 0)	Precision (Class 1)	Recall (Class 0)	Recall (Class 1)	F1-Score (class 0)	F1-Score(class 1)	Accuracy	Weighted Avg Precision	Weighted Avg Recall	Weighted Avg F1-score
Logistic Regression	0.96	0.13	0.89	0.89	0.93	0.19	0.87	0.93	0.87	0.89
Random Forest	0.95	0.00	0.00	0.98	0.97	0.00	0.94	0.91	0.94	0.92
Neural Network	0.95	0.00	0.97	0.00	0.96	0.00	0.92	0.91	0.92	0.92

# Evaluation Metrics Analysis
Logistic Regression with SMOTE
•	Improvement in Minority Class Detection: There's a noticeable improvement in detecting the minority class (class 1) with a recall of 0.33, meaning the model correctly identified 33% of the poor health cases after applying SMOTE. This is a significant improvement from the previous 0% recall.
•	Decrease in Overall Accuracy: The accuracy dropped to 0.87 from 0.95, indicating a trade-off between improving minority class detection and overall accuracy.
•	Increased Precision for Minority Class: The precision for class 1 increased, indicating that among the instances predicted as poor health, 13% were correct.

Random Forest and Neural Network with SMOTE
•	No Improvement in Minority Class Detection: Both models did not show improvement in identifying the minority class after applying SMOTE, with precision, recall, and F1-score for class 1 remaining at 0.
•	High Accuracy for Majority Class: Both models maintained high accuracy, mainly driven by their ability to predict the majority class (class 0) correctly.


The application of SMOTE aimed to improve the models' ability to detect cases of poor physical health, 
Logistic regression demonstrated that SMOTE could help improve the detection of poor health cases at the expense of overall accuracy. 

Random Forest and Neural Network, with the lack of improvement in detecting the minority class after SMOTE, suggests that these models might be too rigid or not effectively leverage the synthetic samples generated by SMOTE. This highlights the challenge of model selection and tuning in imbalanced datasets and the need for further experimentation with model parameters, alternative resampling techniques, or the use of more complex model architecture.

# CONCLUSION

The Logistic Regression model, with its improved ability to identify poor health instances after applying SMOTE, suggests that balancing the class distribution can enhance model sensitivity towards the minority class, aligning with the primary goal of predicting physical health status accurately.
The unchanged performance of the Random Forest and Neural Network models indicates that simply applying SMOTE may not be sufficient to improve minority class detection in all cases. It underscores the importance of exploring a combination of techniques, including different resampling methods, cost-sensitive learning, and advanced modeling approaches.
Given the critical nature of healthcare predictions, models that can more accurately identify individuals at risk of poor health even with slightly lower overall accuracy may be preferred for their potential to impact patient outcomes positively.
