from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when
import sys
import time

def compute_metrics(predictions):
    # Calculate TP, FP, FN
    tp_1 = predictions.filter((col("prediction") == 1.0) & (col("Class") == 1.0)).count()
    fp_1 = predictions.filter((col("prediction") == 1.0) & (col("Class") == 0.0)).count()
    fn_1 = predictions.filter((col("prediction") == 0.0) & (col("Class") == 1.0)).count()
    
    # Calculate Precision, Recall, F1 Score
    precision = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) != 0 else 0.0
    recall = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) != 0 else 0.0
    f1_score = (2.0 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    
    # Calculate accuracy
    correct = predictions.filter(col("prediction") == col("Class")).count()
    total = predictions.count()
    accuracy = correct / total if total != 0 else 0.0
    
    return accuracy, precision, recall, f1_score

def RunLogRegression(input_file, output_file):
    spark = SparkSession.builder.appName("LogisticRegression_StructuredAPI").getOrCreate()
    
    # Read the data
    data = spark.read.csv(input_file, header=True, inferSchema=True)
    input_cols = [col for col in data.columns if col != "Class" and col != "Time"]
    data = data.na.drop()

    # Assemble features
    assembler = VectorAssembler(inputCols=input_cols, outputCol="raw_features")
    assembled_data = assembler.transform(data).select("raw_features", "Class")

    # Split the data into training and test sets
    raw_train, raw_valid, raw_test = assembled_data.randomSplit([0.7, 0.1, 0.2], seed=42)
    
    # Scale the features
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    
    # Scale the training and test set
    scaler_model = scaler.fit(raw_train)
    train = scaler_model.transform(raw_train).select("features", "Class")
    test = scaler_model.transform(raw_test).select("features", "Class")

    # Train the Logistic Regression model
    lr = LogisticRegression(featuresCol="features", labelCol="Class", maxIter=10)
    
    # Training time
    start_time = time.time()
    model = lr.fit(train)
    end_time = time.time()
    training_time = end_time - start_time
    
    # Metric summary
    summary = model.summary
    
    # Predict on test data
    predictions = model.transform(test)

    # Calculate AUC-ROC using BinaryClassificationEvaluator
    auc_evaluator = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC")
    auc = auc_evaluator.evaluate(predictions)
    
    # Evaluate model on test data
    accuracy, precision, recall, f1_score = compute_metrics(predictions)
    
    results = [
        ["Coefficients:", model.coefficients.toArray().tolist()],
        ["Intercept:", model.intercept],
        ["Training Time (seconds):", training_time],
        ["Accuracy (Training):", summary.accuracy],
        ["Accuracy (Test):", accuracy],
        ["Area Under ROC (Training):", summary.areaUnderROC],
        ["Area Under ROC (Test):", auc],
        ["Precision (Test, Class 1):", precision],
        ["Recall (Test, Class 1):", recall],
        ["F1-Score (Test, Class 1):", f1_score]
    ]
    
    # Save best model's result
    spark.sparkContext.parallelize(results).coalesce(1).saveAsTextFile(output_file)
    spark.stop()
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: LogisticRegression <input_file> <output_file>")
        sys.exit(-1)
        
    # Get parameters
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    RunLogRegression(input_file, output_file)
