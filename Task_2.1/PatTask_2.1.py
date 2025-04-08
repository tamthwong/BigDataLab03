from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, unix_timestamp, abs, dayofmonth, month, dayofweek, hour, when, row_number, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import sys

def preprocess_train_data(data):
    print("Preprocessing train data...")
    # Store 'id' separately to reintroduce later
    id_data = data.select("id")

    # Calculate time difference and filter (2-minute tolerance)
    data = data.withColumn("pickup_datetime", to_timestamp(col("pickup_datetime")))
    data = data.withColumn("dropoff_datetime", to_timestamp(col("dropoff_datetime")))
    data = data.withColumn(
        "calculated_duration",
        (unix_timestamp("dropoff_datetime") - unix_timestamp("pickup_datetime"))
    )
    data = data.filter(abs(col("trip_duration") - col("calculated_duration")) <= 120)

    # Drop 'dropoff_datetime' and 'calculated_duration'
    data = data.drop("dropoff_datetime", "calculated_duration")

    # Extract features from 'pickup_datetime'
    data = data.withColumn("day_of_month", dayofmonth(col("pickup_datetime"))) \
               .withColumn("month", month(col("pickup_datetime"))) \
               .withColumn("day_of_week", dayofweek(col("pickup_datetime"))) \
               .withColumn("hour", hour(col("pickup_datetime")))

    # Encode 'store_and_fwd_flag' as 1/0
    data = data.withColumn("store_and_fwd_flag",
                          when(col("store_and_fwd_flag").isin("Y", "yes"), 1).otherwise(0))

    # Define feature columns (excluding 'id')
    feature_cols = ["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude",
                    "dropoff_longitude", "dropoff_latitude", "day_of_month",
                    "month", "day_of_week", "hour", "store_and_fwd_flag"]

    # Use VectorAssembler to combine features into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_data = assembler.transform(data).na.drop()

    print("Train preprocessing complete.")
    return assembled_data, feature_cols, id_data

def preprocess_test_data(data, feature_cols):
    print("Preprocessing test data...")
    # Store 'id' with an index to preserve order
    window = Window.orderBy("id")  # Assuming 'id' order matches test.csv; adjust if needed
    id_data = data.select("id").withColumn("row_index", row_number().over(window))

    # Extract features from 'pickup_datetime'
    data = data.withColumn("pickup_datetime", to_timestamp(col("pickup_datetime")))
    data = data.withColumn("day_of_month", dayofmonth(col("pickup_datetime"))) \
               .withColumn("month", month(col("pickup_datetime"))) \
               .withColumn("day_of_week", dayofweek(col("pickup_datetime"))) \
               .withColumn("hour", hour(col("pickup_datetime")))

    # Encode 'store_and_fwd_flag' as 1/0
    data = data.withColumn("store_and_fwd_flag",
                          when(col("store_and_fwd_flag").isin("Y", "yes"), 1).otherwise(0))

    # Use VectorAssembler with the common feature columns from train.csv
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_data = assembler.transform(data).na.drop().withColumn("row_index", row_number().over(window))

    print("Test preprocessing complete.")
    return assembled_data, id_data

def process_and_train_decision_tree(spark, train_input_path, test_input_path, output_path):
    # Load and preprocess training dataset
    train_data = spark.read.csv(train_input_path, header=True, inferSchema=True)
    train_assembled, feature_cols, train_ids = preprocess_train_data(train_data)

    # Select only features and label for training (exclude 'id')
    train_final = train_assembled.select("features", col("trip_duration").alias("label"))

    # Train-test split for evaluation
    train_split, test_split = train_final.randomSplit([0.8, 0.2], seed=42)

    # Train the Decision Tree Regressor with maxDepth=5
    print("Training model...")
    dt = DecisionTreeRegressor(featuresCol="features", labelCol="label", maxDepth=10, impurity="variance")
    model = dt.fit(train_split)

    # Evaluate on the test split from train.csv (validation split)
    train_predictions = model.transform(test_split)
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
    valid_rmse = evaluator.evaluate(train_predictions, {evaluator.metricName: "rmse"})
    valid_r2 = evaluator.evaluate(train_predictions, {evaluator.metricName: "r2"})
    print("RMSE (validation split):", valid_rmse)
    print("R² (validation split):", valid_r2)

    # Reintroduce 'id' for validation output (optional display)
    train_ids_with_index = train_ids.withColumn("index", monotonically_increasing_id())
    train_predictions_with_index = train_predictions.withColumn("index", monotonically_increasing_id())
    train_predictions_with_id = train_predictions_with_index.join(train_ids_with_index, "index", "inner").drop("index")
    train_predictions_with_id.select("id", "label", "prediction").show(5)

    # 1. Write validation metrics to HDFS
    validation_metrics = spark.createDataFrame(
        [(valid_rmse, valid_r2)],
        ["Valid_RMSE", "Valid_R2"]
    )
    validation_metrics.coalesce(1) \
                      .write \
                      .mode("overwrite") \
                      .csv(f"{output_path}/validation_metrics", header=True)
    print(f"Validation metrics saved to: {output_path}/validation_metrics")

    # 2. Write tree structure and feature importances to HDFS
    tree_structure = model.toDebugString
    feature_importances = model.featureImportances.toArray().tolist()
    feature_importance_dict = dict(zip(feature_cols, feature_importances))
    model_details = [
        ("Tree Structure", tree_structure),
        ("Feature Importances", str(feature_importance_dict))
    ]
    model_details_df = spark.createDataFrame(model_details, ["Metric", "Value"])
    model_details_df.coalesce(1) \
                    .write \
                    .mode("overwrite") \
                    .csv(f"{output_path}/model_details", header=True)
    print("Tree Structure:\n", tree_structure)
    print("Feature Importances:", feature_importances)
    print(f"Model details saved to: {output_path}/model_details")

    # Load and preprocess the test dataset
    test_data = spark.read.csv(test_input_path, header=True, inferSchema=True)
    test_assembled, test_ids = preprocess_test_data(test_data, feature_cols)

    # Select only features for prediction (exclude 'id')
    test_final = test_assembled.select("features", "row_index")

    # Make predictions on test.csv
    print("Making predictions on test data...")
    test_predictions = model.transform(test_final)

    # Join predictions with 'id' using the preserved row_index, exclude 'features'
    test_predictions_with_id = test_predictions.join(test_ids, "row_index", "inner") \
                                               .select("id", "prediction")
    test_predictions_with_id.show(5)

    # 3. Write test predictions to HDFS (only 'id' and 'prediction')
    print("Writing predictions to CSV...")
    test_predictions_with_id.coalesce(1) \
                            .write \
                            .mode("overwrite") \
                            .csv(f"{output_path}/predictions", header=True)
    print(f"Test predictions saved to: {output_path}/predictions")

    return model, test_predictions_with_id

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <train_input_path> <test_input_path> <output_path>")
        sys.exit(1)

    train_input_path = sys.argv[1]
    test_input_path = sys.argv[2]
    output_path = sys.argv[3]

    # Start Spark session
    spark = SparkSession.builder.appName("NYCTaxiTripRegression").getOrCreate()

    try:
        model, test_predictions = process_and_train_decision_tree(spark, train_input_path, test_input_path, output_path)
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down Spark...")
    finally:
        spark.stop()
