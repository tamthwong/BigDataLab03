from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, unix_timestamp, abs, dayofmonth, month, dayofweek, hour, when, log1p, exp, sqrt, sin, cos, asin, radians
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import sys
import time

def preprocess_train_data(spark, data):
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

    # Loại bỏ outliers trong trip_duration (< 60 giây, > 6 giờ)
    data = data.filter((col("trip_duration") > 60) & (col("trip_duration") < 6 * 3600))

    # Lọc tọa độ ngoài phạm vi NYC
    data = data.filter((col("pickup_longitude") > -74.3) & (col("pickup_longitude") < -73.7) &
                       (col("pickup_latitude") > 40.5) & (col("pickup_latitude") < 41.0) &
                       (col("dropoff_longitude") > -74.3) & (col("dropoff_longitude") < -73.7) &
                       (col("dropoff_latitude") > 40.5) & (col("dropoff_latitude") < 41.0))

    # Extract features from 'pickup_datetime'
    data = data.withColumn("day_of_month", dayofmonth(col("pickup_datetime"))) \
               .withColumn("month", month(col("pickup_datetime"))) \
               .withColumn("day_of_week", dayofweek(col("pickup_datetime"))) \
               .withColumn("hour", hour(col("pickup_datetime")))

    # Tạo cờ giờ cao điểm (peak_hour): 7-9h sáng và 4-6h chiều
    data = data.withColumn("peak_hour",
                           when(col("hour").isin(7, 8, 9, 16, 17, 18), 1).otherwise(0))

    # Encode 'store_and_fwd_flag' as 1/0
    data = data.withColumn("store_and_fwd_flag",
                           when(col("store_and_fwd_flag").isin("Y", "yes"), 1).otherwise(0))

    # Tính khoảng cách Haversine
    # Đổi các cột thành radian trước
    data = data.withColumn("lat1_rad", radians(col("pickup_latitude"))) \
            .withColumn("lon1_rad", radians(col("pickup_longitude"))) \
            .withColumn("lat2_rad", radians(col("dropoff_latitude"))) \
            .withColumn("lon2_rad", radians(col("dropoff_longitude")))

    # Tính toán các phần trong công thức Haversine
    data = data.withColumn("dlat", col("lat2_rad") - col("lat1_rad")) \
            .withColumn("dlon", col("lon2_rad") - col("lon1_rad")) \
            .withColumn("a", sin(col("dlat")/2)**2 +
                                cos(col("lat1_rad")) * cos(col("lat2_rad")) *
                                sin(col("dlon")/2)**2) \
            .withColumn("c", 2 * asin(sqrt(col("a")))) \
            .withColumn("distance_haversine", col("c") * 6371)

    # Tính tốc độ trung bình (km/h) và lọc outliers (> 100 km/h)
    data = data.withColumn("speed", col("distance_haversine") / (col("trip_duration") / 3600))
    data = data.filter(col("speed") < 100)

    # Define feature columns (including new features)
    feature_cols = ["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude",
                    "dropoff_longitude", "dropoff_latitude", "day_of_month",
                    "month", "day_of_week", "hour", "store_and_fwd_flag", "peak_hour", "distance_haversine"]

    # Use VectorAssembler to combine features into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_data = assembler.transform(data).na.drop()

    # Log transform trip_duration
    assembled_data = assembled_data.withColumn("label", log1p(col("trip_duration")))

    print("Train preprocessing complete.")
    return assembled_data, feature_cols, id_data

def preprocess_test_data(spark, data, feature_cols):
    print("Preprocessing test data...")
    # Store 'id' to reintroduce later
    id_data = data.select("id")

    # Extract features from 'pickup_datetime'
    data = data.withColumn("pickup_datetime", to_timestamp(col("pickup_datetime"))) \
               .withColumn("day_of_month", dayofmonth(col("pickup_datetime"))) \
               .withColumn("month", month(col("pickup_datetime"))) \
               .withColumn("day_of_week", dayofweek(col("pickup_datetime"))) \
               .withColumn("hour", hour(col("pickup_datetime")))

    # Tạo cờ giờ cao điểm (peak_hour): 7-9h sáng và 4-6h chiều
    data = data.withColumn("peak_hour",
                           when(col("hour").isin(7, 8, 9, 16, 17, 18), 1).otherwise(0))

    # Encode 'store_and_fwd_flag' as 1/0
    data = data.withColumn("store_and_fwd_flag",
                           when(col("store_and_fwd_flag").isin("Y", "yes"), 1).otherwise(0))

    data = data.withColumn("lat1_rad", radians(col("pickup_latitude"))) \
            .withColumn("lon1_rad", radians(col("pickup_longitude"))) \
            .withColumn("lat2_rad", radians(col("dropoff_latitude"))) \
            .withColumn("lon2_rad", radians(col("dropoff_longitude")))

    # Calculate parts of Haversine formula
    data = data.withColumn("dlat", col("lat2_rad") - col("lat1_rad")) \
            .withColumn("dlon", col("lon2_rad") - col("lon1_rad")) \
            .withColumn("a", sin(col("dlat")/2)**2 +
                                cos(col("lat1_rad")) * cos(col("lat2_rad")) *
                                sin(col("dlon")/2)**2) \
            .withColumn("c", 2 * asin(sqrt(col("a")))) \
            .withColumn("distance_haversine", col("c") * 6371)

    # Use VectorAssembler with the common feature columns from train.csv
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_data = assembler.transform(data)

    # Join back with id_data to keep only the rows that survived na.drop()
    assembled_data = assembled_data.join(id_data, "id", "inner")

    print("Test preprocessing complete.")
    return assembled_data

def process_and_train_decision_tree(spark, train_input_path, test_input_path, output_path):
    # Start measuring total processing time
    start_time = time.time()

    # Load and preprocess training dataset
    train_data = spark.read.csv(train_input_path, header=True, inferSchema=True)
    train_assembled, feature_cols, train_ids = preprocess_train_data(spark, train_data)

    # Select only features and label for training (exclude 'id')
    train_final = train_assembled.select("features", "label")

    # Train-test split for evaluation
    train_split, test_split = train_final.randomSplit([0.8, 0.2], seed=42)

    # GridSearch
    max_depths = [i for i in range(1, 15)]
    impurity = "variance"  # Fixed for regression
    best_rmse = float("inf")
    best_r2 = float("inf")
    best_model = None
    best_max_depth = None
    best_training_time = None

    # Iterate over maxDepth values
    print("Starting hyperparameter tuning...")
    for max_depth in max_depths:
        print(f"--- Training with depth {max_depth} ---")
        train_start_time = time.time()
        dt = DecisionTreeRegressor(featuresCol="features", labelCol="label", maxDepth=max_depth, impurity=impurity)
        model = dt.fit(train_split)
        train_end_time = time.time()
        training_time = train_end_time - train_start_time
        print(f"Training time for maxDepth={max_depth}: {training_time:.2f} seconds")

        # Evaluate on the validation split
        train_predictions = model.transform(test_split)
        # Chuyển ngược log transform để tính RMSE và R² trên thang gốc
        train_predictions = train_predictions.withColumn("prediction_original", exp(col("prediction")) - 1)
        train_predictions = train_predictions.withColumn("label_original", exp(col("label")) - 1)

        evaluator = RegressionEvaluator(labelCol="label_original", predictionCol="prediction_original")
        valid_rmse = evaluator.evaluate(train_predictions, {evaluator.metricName: "rmse"})
        valid_r2 = evaluator.evaluate(train_predictions, {evaluator.metricName: "r2"})
        print(f"RMSE (validation split, original scale) for maxDepth={max_depth}: {valid_rmse}")
        print(f"R² (validation split, original scale) for maxDepth={max_depth}: {valid_r2}")

        # Update best model if this RMSE is lower
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_r2 = valid_r2
            best_model = model
            best_max_depth = max_depth
            best_training_time = training_time

    print(f"\nBest model: maxDepth={best_max_depth}, impurity={impurity}")
    print(f"Best RMSE: {best_rmse}")
    print(f"Best R²: {best_r2}")
    print(f"Training time for best model: {best_training_time:.2f} seconds")

    # Load and preprocess the test dataset
    test_data = spark.read.csv(test_input_path, header=True, inferSchema=True)
    test_assembled = preprocess_test_data(spark, test_data, feature_cols)

    # Select features and id for prediction
    test_final = test_assembled.select("id", "features")

    # Make predictions on test.csv using the best model
    print("Making predictions on test data with best model...")
    test_predictions = best_model.transform(test_final)
    # Chuyển ngược log transform cho kết quả dự đoán
    test_predictions = test_predictions.withColumn("prediction", exp(col("prediction")) - 1)

    # Select only 'id' and 'prediction', sort by 'id'
    test_predictions_with_id = test_predictions.select("id", "prediction").orderBy("id")
    test_predictions_with_id.show(5)

    # Write test predictions to HDFS (only 'id' and 'prediction')
    print("Writing predictions to CSV...")
    test_predictions_with_id.coalesce(1) \
                            .write \
                            .mode("overwrite") \
                            .csv(f"{output_path}/predictions", header=True)
    print(f"Test predictions saved to: {output_path}/predictions")

    # Calculate total processing time
    end_time = time.time()
    total_processing_time = end_time - start_time
    print(f"Total processing time: {total_processing_time:.2f} seconds")

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
        ("Impurity", impurity),
        ("Training Time (Seconds)", str(best_training_time)),
        ("Total Processing Time (Seconds)", str(total_processing_time)),
    ]
    model_metrics_df = spark.createDataFrame(model_metrics, ["Metric", "Value"])
    model_metrics_df.coalesce(1) \
                    .write \
                    .mode("overwrite") \
                    .csv(f"{output_path}/model_metrics", header=True)
    print(f"All metrics saved to: {output_path}/model_metrics")

    return best_model, test_predictions_with_id, best_training_time, total_processing_time

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
        model, test_predictions, training_time, total_processing_time = process_and_train_decision_tree(
            spark, train_input_path, test_input_path, output_path
        )
        print("\nSummary of Performance Metrics:")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Total Processing Time: {total_processing_time:.2f} seconds")
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down Spark...")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
