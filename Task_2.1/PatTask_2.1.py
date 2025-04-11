from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, unix_timestamp, abs, dayofmonth, month, dayofweek, hour, when, monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import sys
import time

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
    # Store 'id' to reintroduce later (no need for row_index since we're sorting)
    id_data = data.select("id")

    # Extract features from 'pickup_datetime'
    data = data.withColumn("pickup_datetime", to_timestamp(col("pickup_datetime"))) \
               .withColumn("day_of_month", dayofmonth(col("pickup_datetime"))) \
               .withColumn("month", month(col("pickup_datetime"))) \
               .withColumn("day_of_week", dayofweek(col("pickup_datetime"))) \
               .withColumn("hour", hour(col("pickup_datetime")))

    # Encode 'store_and_fwd_flag' as 1/0
    data = data.withColumn("store_and_fwd_flag",
                          when(col("store_and_fwd_flag").isin("Y", "yes"), 1).otherwise(0))

    # Use VectorAssembler with the common feature columns from train.csv
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_data = assembler.transform(data)

    # Apply na.drop()
    assembled_data = assembled_data.na.drop()

    # Join back with id_data to keep only the rows that survived na.drop()
    assembled_data = assembled_data.join(id_data, "id", "inner")

    print("Test preprocessing complete.")
    return assembled_data

def process_and_train_decision_tree(spark, train_input_path, test_input_path, output_path):
    # Load and preprocess training dataset
    train_data = spark.read.csv(train_input_path, header=True, inferSchema=True)
    train_assembled, feature_cols, train_ids = preprocess_train_data(train_data)

    # Select only features and label for training (exclude 'id')
    train_final = train_assembled.select("features", col("trip_duration").alias("label"))

    # Train-test split for evaluation
    train_split, test_split = train_final.randomSplit([0.8, 0.2], seed=42)

    # GridSearch
    max_depths = [i for i in range(1, 11)]
    best_rmse = float("inf")
    best_r2 = float("inf")
    best_model = None
    best_max_depth = None
    training_time = 0

    # Train the Decision Tree Regressor with maxDepth
    for max_depth in max_depths:
        print(f"--- Training with depth {max_depth} ---")

        print("Training model...")
        train_start_time = time.time()
        dt = DecisionTreeRegressor(featuresCol="features", labelCol="label", maxDepth=max_depth, impurity="variance")
        model = dt.fit(train_split)
        train_end_time = time.time()
        training_time = train_end_time - train_start_time
        print(f"Training time: {training_time:.2f} seconds")

        # Evaluate on the test split from train.csv (validation split)
        train_predictions = model.transform(test_split)
        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
        valid_rmse = evaluator.evaluate(train_predictions, {evaluator.metricName: "rmse"})
        valid_r2 = evaluator.evaluate(train_predictions, {evaluator.metricName: "r2"})
        print("RMSE (validation split):", valid_rmse)
        print("R² (validation split):", valid_r2)

        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_r2 = valid_r2
            best_model = model
            best_max_depth = max_depth

    # Load and preprocess the test dataset
    test_data = spark.read.csv(test_input_path, header=True, inferSchema=True)
    test_assembled = preprocess_test_data(test_data, feature_cols)

    # Select features and id for prediction
    test_final = test_assembled.select("id", "features")

    # Make predictions on test.csv
    print("Making predictions on test data...")
    test_predictions = best_model.transform(test_final)

    # Select only 'id' and 'prediction', sort by 'id'
    test_predictions_with_id = test_predictions.select("id", "prediction").orderBy("id")

    # Write test predictions to HDFS (only 'id' and 'prediction')
    print("Writing predictions to CSV...")
    test_predictions_with_id.coalesce(1) \
                            .write \
                            .mode("overwrite") \
                            .csv(f"{output_path}/predictions", header=True)
    print(f"Test predictions saved to: {output_path}/predictions")

    # Combine all metrics into a single file
    tree_structure = best_model.toDebugString
    feature_importances = best_model.featureImportances.toArray().tolist()
    feature_importance_dict = dict(zip(feature_cols, feature_importances))
    model_metrics = [
        ("Tree", tree_structure),
        ("Features Importance", str(feature_importance_dict)),
        ("Valid RMSE", str(best_rmse)),
        ("Valid R2", str(best_r2)),
        ("Best Max Depth", str(best_max_depth)),
        ("Training Time (Seconds)", str(training_time))
    ]
    model_metrics_df = spark.createDataFrame(model_metrics, ["Metric", "Value"])
    model_metrics_df.coalesce(1) \
                    .write \
                    .mode("overwrite") \
                    .csv(f"{output_path}/model_metrics", header=True)
    print(f"All metrics saved to: {output_path}/model_metrics")

    return best_model, test_predictions_with_id, training_time

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <train_input_path> <test_input_path> <output_path>")
        sys.exit(1)

    train_input_path = sys.argv[1]
    test_input_path = sys.argv[2]
    output_path = sys.argv[3]

    # Start Spark session
    spark = SparkSession.builder.appName("NYCTaxiTripRegression").getOrCreate()

    try:
        model, test_predictions, training_time = process_and_train_decision_tree(
            spark, train_input_path, test_input_path, output_path
        )
        print("\nSummary of Performance Metrics:")
        print(f"Training Time: {training_time:.2f} seconds")
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down Spark...")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
