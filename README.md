# Task 1: Classification with Logistic Regression
For both 3 task 3.1.1; 3.1.2; 3.1.3   
The program receives two command-line arguments: the path to the creditcard.csv dataset and the path to the output directory.
For example: 
```bash
spark-submit –master yarn \
LogisticRegression_StructuredAPI.py \
/path/to/creditcard.csv \
/path/to/output
```
The output file contains the model’s coefficients and evaluation metrics, including accuracy, AUC (Area Under the ROC Curve), precision, and recall.

# Task 2: Regression with Decision Trees

For both 3 task: 3.2.1; 3.2.2; 3.2.3  
The program receives three command-line arguments: the path to the train.csv file, the path to the test.csv file, and the path to the output directory.
For example: 
```bash
spark-submit –master yarn \
TreeClassification_StructuredAPI.py \
/path/to/train.csv \
/path/to/test.csv \
/path/to/output
```
The output folder contains two subfolder: **model_results** and **predictions**. The model_results subfolder includes the best tree structure, its corresponding hyperparameters such as maximum depth and number of bins, as well as evaluation metrics like **RMSE** and **R²**. The predictions subfolder stores the model's predictions on the test set, including two columns: the ID of each data row and its corresponding predicted value.
