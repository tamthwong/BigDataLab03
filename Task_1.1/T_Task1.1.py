from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import sys

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
    
    predictions = model.transform(test)

    auc_evaluator = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC")
    auc = auc_evaluator.evaluate(predictions)

    acc_evaluator = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction", metricName="accuracy")
    accuracy = acc_evaluator.evaluate(predictions)

    precision_evaluator = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction", metricName="weightedPrecision")
    precision = precision_evaluator.evaluate(predictions)

    recall_evaluator = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction", metricName="weightedRecall")
    recall = recall_evaluator.evaluate(predictions)
    
    results = [
        ["Coefficients:", model.coefficients.toArray().tolist()],
        ["Intercept:", model.intercept],
        ["Accuracy (Training):", summary.accuracy],
        ["Accuracy (Test):", accuracy],
        ["Area Under ROC (Training):", summary.areaUnderROC],
        ["Area Under ROC (Test):", auc],
        ["Precision (Test):", precision],
        ["Recall (Test):", recall]
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