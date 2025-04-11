from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when
import sys

def compute_metrics(predictions):
    # Tính TP, FP, FN cho lớp 0 và lớp 1 sử dụng DataFrame
    tp_0 = predictions.filter((col("prediction") == 0.0) & (col("Class") == 0.0)).count()
    fp_0 = predictions.filter((col("prediction") == 0.0) & (col("Class") == 1.0)).count()
    fn_0 = predictions.filter((col("prediction") == 1.0) & (col("Class") == 0.0)).count()
    
    tp_1 = predictions.filter((col("prediction") == 1.0) & (col("Class") == 1.0)).count()
    fp_1 = predictions.filter((col("prediction") == 1.0) & (col("Class") == 0.0)).count()
    fn_1 = predictions.filter((col("prediction") == 0.0) & (col("Class") == 1.0)).count()
    
    # Tính precision và recall cho lớp 0
    precision_0 = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) != 0 else 0.0
    recall_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) != 0 else 0.0
    
    # Tính precision và recall cho lớp 1
    precision_1 = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) != 0 else 0.0
    recall_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) != 0 else 0.0
    
    # Tính unweighted (macro-average) precision và recall
    unweighted_precision = (precision_0 + precision_1) / 2.0
    unweighted_recall = (recall_0 + recall_1) / 2.0
    
    # Tính unweighted F1-score
    unweighted_f1 = (2.0 * unweighted_precision * unweighted_recall) / (unweighted_precision + unweighted_recall) if (unweighted_precision + unweighted_recall) != 0 else 0.0
    
    # Tính accuracy
    correct = predictions.filter(col("prediction") == col("Class")).count()
    total = predictions.count()
    accuracy = correct / total if total != 0 else 0.0
    
    return accuracy, unweighted_precision, unweighted_recall, unweighted_f1

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
    
    # Scale the training set
    scaler_model = scaler.fit(raw_train)
    train = scaler_model.transform(raw_train).select("features", "Class")

    # Scale the test set
    test = scaler_model.transform(raw_test).select("features", "Class")

    # Train the Logistic Regression model
    lr = LogisticRegression(featuresCol="features", labelCol="Class", maxIter=10)
    model = lr.fit(train)

    summary = model.summary
    
    # Dự đoán trên tập test
    predictions = model.transform(test)

    # Tính AUC bằng BinaryClassificationEvaluator
    auc_evaluator = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC")
    auc = auc_evaluator.evaluate(predictions)
    
    # Tính các chỉ số unweighted
    accuracy, precision, recall, f1_score = compute_metrics(predictions)
    
    results = [
        ["Coefficients:", model.coefficients.toArray().tolist()],
        ["Intercept:", model.intercept],
        ["Accuracy (Training):", summary.accuracy],
        ["Accuracy (Test):", accuracy],
        ["Area Under ROC (Training):", summary.areaUnderROC],
        ["Area Under ROC (Test):", auc],
        ["Precision (Test, Unweighted):", precision],
        ["Recall (Test, Unweighted):", recall],
        ["F1-Score (Test, Unweighted):", f1_score]
    ]
    
    spark.sparkContext.parallelize(results).coalesce(1).saveAsTextFile(output_file)
    spark.stop()
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: LogisticRegression <input_file> <output_file>")
        sys.exit(-1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    RunLogRegression(input_file, output_file)