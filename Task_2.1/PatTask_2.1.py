from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, unix_timestamp, abs, dayofmonth, month, dayofweek, hour, when, \
    log1p, exp, sqrt, sin, cos, asin, radians
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import sys
import time

def preprocess_train_data(spark, data):
    """
    Preprocess training data by cleaning, filtering outliers, and extracting features.

    Args:
        spark: SparkSession object
        data: Input DataFrame containing training data

    Returns:
        Tuple of (assembled_data, feature_cols, id_data) where assembled_data is the processed DataFrame,
        feature_cols is the list of feature names, and id_data is the DataFrame with 'id' column
    """
    print("Preprocessing train data...")

    # Preserve 'id' column for later use
    id_data = data.select("id")

    # Convert datetime columns to timestamp format
    data = data.withColumn("pickup_datetime", to_timestamp(col("pickup_datetime")))
    data = data.withColumn("dropoff_datetime", to_timestamp(col("dropoff_datetime")))
    
    # Calculate trip duration from timestamps (in seconds)
    data = data.withColumn(
        "calculated_duration",
        (unix_timestamp("dropoff_datetime") - unix_timestamp("pickup_datetime"))
    )

    # Filter trips where recorded duration is within 2 minutes of calculated duration
    data = data.filter(abs(col("trip_duration") - col("calculated_duration")) <= 120)

    # Drop temporary columns as they are no longer needed
    data = data.drop("dropoff_datetime", "calculated_duration")

    # Exclude trips with unrealistic durations (< 60s or > 6h)
    data = data.filter((col("trip_duration") > 60) & (col("trip_duration") < 6 * 3600))

    # Restrict coordinates to New York City geographical bounds
    data = data.filter(
        (col("pickup_longitude") > -74.3) & (col("pickup_longitude") < -73.7) &
        (col("pickup_latitude") > 40.5) & (col("pickup_latitude") < 41.0) &
        (col("dropoff_longitude") > -74.3) & (col("dropoff_longitude") < -73.7) &
        (col("dropoff_latitude") > 40.5) & (col("dropoff_latitude") < 41.0)
    )

    # Extract temporal features from pickup datetime
    data = data.withColumn("day_of_month", dayofmonth(col("pickup_datetime"))) \
               .withColumn("month", month(col("pickup_datetime"))) \
               .withColumn("day_of_week", dayofweek(col("pickup_datetime"))) \
               .withColumn("hour", hour(col("pickup_datetime")))

    # Create binary flag for peak hours (7-9 AM, 4-6 PM)
    data = data.withColumn(
        "peak_hour",
        when(col("hour").isin(7, 8, 9, 16, 17, 18), 1).otherwise(0)
    )

    # Encode store_and_fwd_flag as binary (1 for 'Y'/'yes', 0 otherwise)
    data = data.withColumn(
        "store_and_fwd_flag",
        when(col("store_and_fwd_flag").isin("Y", "yes"), 1).otherwise(0)
    )

    # Convert latitude and longitude to radians for distance calculation
    data = data.withColumn("lat1_rad", radians(col("pickup_latitude"))) \
               .withColumn("lon1_rad", radians(col("pickup_longitude"))) \
               .withColumn("lat2_rad", radians(col("dropoff_latitude"))) \
               .withColumn("lon2_rad", radians(col("dropoff_longitude")))

    # Calculate Haversine distance (in km) between pickup and dropoff
    data = data.withColumn("dlat", col("lat2_rad") - col("lat1_rad")) \
               .withColumn("dlon", col("lon2_rad") - col("lon1_rad")) \
               .withColumn(
                   "a",
                   sin(col("dlat") / 2) ** 2 +
                   cos(col("lat1_rad")) * cos(col("lat2_rad")) * sin(col("dlon") / 2) ** 2
               ) \
               .withColumn("c", 2 * asin(sqrt(col("a")))) \
               .withColumn("distance_haversine", col("c") * 6371)

    # Compute average speed (km/h) and filter out trips exceeding 100 km/h
    data = data.withColumn("speed", col("distance_haversine") / (col("trip_duration") / 3600))
    data = data.filter(col("speed") < 100)

    # List of feature columns for the model
    feature_cols = [
        "vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "day_of_month",
        "month", "day_of_week", "hour", "store_and_fwd_flag", "peak_hour", "distance_haversine"
    ]

    # Combine features into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_data = assembler.transform(data).na.drop()

    # Transform target variable (trip_duration) using log1p for better model fit
    assembled_data = assembled_data.withColumn("label", log1p(col("trip_duration")))

    print("Train preprocessing complete.")
    return assembled_data, feature_cols, id_data

def preprocess_test_data(spark, data, feature_cols):
    """
    Preprocess test data to align with training data features.

    Args:
        spark: SparkSession object
        data: Input DataFrame containing test data
        feature_cols: List of feature column names from training

    Returns:
        Processed DataFrame with features and 'id' column
    """
    print("Preprocessing test data...")

    # Preserve 'id' column for output
    id_data = data.select("id")

    # Extract temporal features from pickup datetime
    data = data.withColumn("pickup_datetime", to_timestamp(col("pickup_datetime"))) \
               .withColumn("day_of_month", dayofmonth(col("pickup_datetime"))) \
               .withColumn("month", month(col("pickup_datetime"))) \
               .withColumn("day_of_week", dayofweek(col("pickup_datetime"))) \
               .withColumn("hour", hour(col("pickup_datetime")))

    # Create binary flag for peak hours
    data = data.withColumn(
        "peak_hour",
        when(col("hour").isin(7, 8, 9, 16, 17, 18), 1).otherwise(0)
    )

    # Encode store_and_fwd_flag as binary
    data = data.withColumn(
        "store_and_fwd_flag",
        when(col("store_and_fwd_flag").isin("Y", "yes"), 1).otherwise(0)
    )

    # Convert coordinates to radians
    data = data.withColumn("lat1_rad", radians(col("pickup_latitude"))) \
               .withColumn("lon1_rad", radians(col("pickup_longitude"))) \
               .withColumn("lat2_rad", radians(col("dropoff_latitude"))) \
               .withColumn("lon2_rad", radians(col("dropoff_longitude")))

    # Calculate Haversine distance
    data = data.withColumn("dlat", col("lat2_rad") - col("lat1_rad")) \
               .withColumn("dlon", col("lon2_rad") - col("lon1_rad")) \
               .withColumn(
                   "a",
                   sin(col("dlat") / 2) ** 2 +
                   cos(col("lat1_rad")) * cos(col("lat2_rad")) * sin(col("dlon") / 2) ** 2
               ) \
               .withColumn("c", 2 * asin(sqrt(col("a")))) \
               .withColumn("distance_haversine", col("c") * 6371)

    # Assemble features into a vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_data = assembler.transform(data)

    print("Test preprocessing complete.")
    return assembled_data

def grid_search(train_split, test_split, max_depths, impurity="variance"):
    """
    Perform grid search over maxDepth to find the best DecisionTreeRegressor model.

    Args:
        train_split: Training DataFrame with features and label
        test_split: Validation DataFrame for evaluation
        max_depths: List of maximum depths to evaluate
        impurity: Impurity measure for tree splits (default: variance)

    Returns:
        Tuple of (best_model, best_rmse, best_r2, best_max_depth, best_training_time)
    """
    best_rmse = float("inf")
    best_r2 = float("-inf")
    best_model = None
    best_max_depth = None
    best_training_time = None

    print("Starting hyperparameter tuning...")
    for max_depth in max_depths:
        print(f"--- Training with maxDepth={max_depth} ---")

        # Start timing the training process
        train_start_time = time.time()

        # Initialize and train the decision tree
        dt = DecisionTreeRegressor(
            featuresCol="features", labelCol="label", maxDepth=max_depth, impurity=impurity
        )
        model = dt.fit(train_split)

        # Calculate training duration
        train_end_time = time.time()
        training_time = train_end_time - train_start_time
        print(f"Training time for maxDepth={max_depth}: {training_time:.2f} seconds")

        # Predict on validation set
        predictions = model.transform(test_split)

        # Convert predictions and labels back to original scale
        predictions = predictions.withColumn("prediction_original", exp(col("prediction")) - 1)
        predictions = predictions.withColumn("label_original", exp(col("label")) - 1)

        # Evaluate performance metrics
        evaluator = RegressionEvaluator(labelCol="label_original", predictionCol="prediction_original")
        valid_rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        valid_r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
        print(f"RMSE (validation, original scale) for maxDepth={max_depth}: {valid_rmse:.2f}")
        print(f"R² (validation, original scale) for maxDepth={max_depth}: {valid_r2:.4f}")

        # Update best model if RMSE improves
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_r2 = valid_r2
            best_model = model
            best_max_depth = max_depth
            best_training_time = training_time

    print(f"\nBest model: maxDepth={best_max_depth}, impurity={impurity}")
    print(f"Best RMSE: {best_rmse:.2f}")
    print(f"Best R²: {best_r2:.4f}")
    print(f"Training time for best model: {best_training_time:.2f} seconds")
    return best_model, best_rmse, best_r2, best_max_depth, best_training_time

def process_and_train_decision_tree(spark, train_input_path, test_input_path, output_path):
    """
    Main function to process data, train model, and generate predictions.

    Args:
        spark: SparkSession object
        train_input_path: Path to training CSV file
        test_input_path: Path to test CSV file
        output_path: Path to save predictions and metrics

    Returns:
        Tuple of (best_model, test_predictions, best_training_time, total_processing_time)
    """
    # Start timing the entire process
    start_time = time.time()

    # Load training data
    train_data = spark.read.csv(train_input_path, header=True, inferSchema=True)

    # Preprocess training data
    train_assembled, feature_cols, train_ids = preprocess_train_data(spark, train_data)

    # Select features and label for training
    train_final = train_assembled.select("features", "label")

    # Split into training and validation sets
    train_split, test_split = train_final.randomSplit([0.8, 0.2], seed=42)

    # Perform grid search to find optimal max_depth
    max_depths = list(range(1, 21))
    best_model, best_rmse, best_r2, best_max_depth, best_training_time = grid_search(
        train_split, test_split, max_depths, impurity="variance"
    )

    # Load test data
    test_data = spark.read.csv(test_input_path, header=True, inferSchema=True)

    # Preprocess test data
    test_assembled = preprocess_test_data(spark, test_data, feature_cols)

    # Select features and id for predictions
    test_final = test_assembled.select("id", "features")

    # Generate predictions on test data
    print("Making predictions on test data with best model...")
    test_predictions = best_model.transform(test_final)

    # Convert predictions to original scale
    test_predictions = test_predictions.withColumn("prediction", exp(col("prediction")) - 1)

    # Select and sort predictions by id
    test_predictions_with_id = test_predictions.select("id", "prediction").orderBy("id")
    test_predictions_with_id.show(5, truncate=False)

    # Save predictions to CSV
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

    # Gather model metrics for analysis
    tree_structure = best_model.toDebugString
    feature_importances = best_model.featureImportances.toArray().tolist()
    feature_importance_dict = dict(zip(feature_cols, feature_importances))
    model_metrics = [
        ("Tree Structure", tree_structure),
        ("Feature Importances", str(feature_importance_dict)),
        ("Validation RMSE", str(best_rmse)),
        ("Validation R2", str(best_r2)),
        ("Best Max Depth", str(best_max_depth)),
        ("Impurity", "variance"),
        ("Training Time (Seconds)", str(best_training_time)),
        ("Total Processing Time (Seconds)", str(total_processing_time)),
    ]

    # Save metrics to CSV
    model_metrics_df = spark.createDataFrame(model_metrics, ["Metric", "Value"])
    model_metrics_df.coalesce(1) \
                   .write \
                   .mode("overwrite") \
                   .csv(f"{output_path}/model_metrics", header=True)
    print(f"Model metrics saved to: {output_path}/model_metrics")

    return best_model, test_predictions_with_id, best_training_time, total_processing_time

def main():
    """
    Entry point for the script, handling input arguments and orchestrating execution.

    Expects command-line arguments: train_input_path, test_input_path, output_path
    """
    if len(sys.argv) != 4:
        print("Usage: python script.py <train_input_path> <test_input_path> <output_path>")
        sys.exit(1)

    train_input_path = sys.argv[1]
    test_input_path = sys.argv[2]
    output_path = sys.argv[3]

    # Initialize Spark session
    spark = SparkSession.builder.appName("NYCTaxiTripRegression").getOrCreate()

    try:
        # Execute data processing and model training
        model, test_predictions, training_time, total_processing_time = process_and_train_decision_tree(
            spark, train_input_path, test_input_path, output_path
        )

        # Print summary of performance
        print("\nSummary of Performance Metrics:")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Total Processing Time: {total_processing_time:.2f} seconds")
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down Spark...")
    finally:
        # Ensure Spark session is properly closed
        spark.stop()

if __name__ == "__main__":
    main()