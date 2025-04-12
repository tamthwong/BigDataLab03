from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, unix_timestamp, col, when, dayofmonth, month, hour, dayofweek, radians, sin, cos, asin, sqrt, log1p, abs
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import RegressionMetrics
import sys
import time
import math

# Evaluate the model using RMSE and R2 with batch processing
def evaluate_model(model, rdd, sc):
    print("Evaluating model...")
    
    try:
        # Predict data
        features_rdd = rdd.map(lambda lp: lp.features)
        labels_rdd = rdd.map(lambda lp: lp.label)
        predictions = model.predict(features_rdd)
        
        # Create result RDD from prediction results and labels
        predictions_and_labels = predictions.zip(labels_rdd).map(
            lambda p_l: (math.exp(float(p_l[0])) - 1, math.exp(float(p_l[1])) - 1)
        )
        
        # Check whether the result RDD is empty
        count = predictions_and_labels.count()
        if count == 0:
            print("Error: RDD is empty")
            return float('inf'), 0.0
        
        # Calculate rmse and r2 metrics
        metrics = RegressionMetrics(predictions_and_labels)
        rmse = metrics.rootMeanSquaredError
        r2 = metrics.r2
        print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")
        return rmse, r2
    
    except Exception as e:
        print(f"Error in evaluate_model: {e}")
        return float('inf'), 0.0

# Get feature importances
def parse_feature_usage(debug_string, feature_cols):
    feature_counts = {col: 0 for col in feature_cols}
    lines = debug_string.split("\n")
    for line in lines:
        for col_idx, col_name in enumerate(feature_cols):
            if f"feature {col_idx}" in line:
                feature_counts[col_name] += 1
    total_splits = sum(feature_counts.values())
    feature_importances = {k: v / total_splits if total_splits > 0 else 0 for k, v in feature_counts.items()}
    return feature_importances

# Read data from csv file and preprocess that data
def read_and_preprocess_data(spark, input_file, isTrain=True):
    data = spark.read.csv(input_file, header=True, inferSchema=True)
    
    # Save id column for later usage
    id_data = data.select("id")
    
    # Convert pickup_datatime to timestamp for later split
    data = data.withColumn("pickup_datetime", to_timestamp(col("pickup_datetime")))
    ### Train data only
    if isTrain:
        data = data.withColumn("dropoff_datetime", to_timestamp(col("dropoff_datetime")))
        
        # Add calculated_duration for validating trip_duration
        data = data.withColumn("calculated_duration", (unix_timestamp("dropoff_datetime") - unix_timestamp("pickup_datetime")))
        
        # If the time difference is greater than 2 minutes, this data is considered as invalid
        data = data.filter(abs(col("trip_duration") - col("calculated_duration")) <= 120)
        data = data.drop("dropoff_datetime", "calculated_duration")
        
        # Validating the data having trip_duration less than 1 minute and greater than 6 hours
        data = data.filter((col("trip_duration") > 60) & (col("trip_duration") < 6 * 3600))
        
        # Filter out coordinates outside New York City
        data = data.filter(
            (col("pickup_longitude").between(-74.3, -73.7)) &
            (col("pickup_latitude").between(40.5, 41.0)) &
            (col("dropoff_longitude").between(-74.3, -73.7)) &
            (col("dropoff_latitude").between(40.5, 41.0))
        )
        
    # Extract features from 'pickup_datetime'
    data = data.withColumn("day_of_month", dayofmonth(col("pickup_datetime"))) \
               .withColumn("month", month(col("pickup_datetime"))) \
               .withColumn("day_of_week", dayofweek(col("pickup_datetime"))) \
               .withColumn("hour", hour(col("pickup_datetime")))
    
    # Create pick-hour flag: 7-9AM and 4-6PM
    data = data.withColumn("peak_hour", when(col("hour").isin(7, 8, 9, 16, 17, 18), 1).otherwise(0))
    
    # Encode 'store_and_fwd_flag' as 1/0
    data = data.withColumn("store_and_fwd_flag", when(col("store_and_fwd_flag").isin("Y", "yes"), 1).otherwise(0))
    
    # Calculate Haversine distance
    data = data.withColumn(
        "distance_haversine",
        2 * asin(sqrt(
            sin((radians(col("dropoff_latitude")) - radians(col("pickup_latitude"))) / 2) ** 2 +
            cos(radians(col("pickup_latitude"))) * cos(radians(col("dropoff_latitude"))) *
            sin((radians(col("dropoff_longitude")) - radians(col("pickup_longitude"))) / 2) ** 2
        )) * 6371
    )
    
    # Calculate average speed (km/h) and filter outliers (> 100 km/h)
    if isTrain:
        data = data.withColumn("speed", col("distance_haversine") / (col("trip_duration") / 3600))
        data = data.filter(col("speed") < 100)
    
    # Define feature columns (including new features)
    feature_cols = ["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude",
                    "dropoff_longitude", "dropoff_latitude", "day_of_month",
                    "month", "day_of_week", "hour", "store_and_fwd_flag", "peak_hour", "distance_haversine"]
    
    # Use VectorAssembler to combine features into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_data = assembler.transform(data).cache()
    
    if isTrain:
        assembled_data = assembled_data.na.drop()
        
        # Log transform trip_duration
        assembled_data = assembled_data.withColumn("label", log1p(col("trip_duration")))
        rdd = assembled_data.select("label", "features").rdd.map(
            lambda row: LabeledPoint(row["label"], row["features"].toArray())
        )
    else:
        assembled_data = assembled_data.join(id_data, "id", "inner")
        rdd = assembled_data.select("id", "features").rdd.map(
            lambda row: (row["id"], row["features"].toArray())
        )
    
    print("Preprocessing complete.")
    return rdd

# Decision Tree Regression
def RDDTreeClassification(spark, train_file, test_file, output_file):
    sc = spark.sparkContext
    
    start_time = time.time()
    
    # Read and preprocess data
    train_rdd = read_and_preprocess_data(spark, train_file, isTrain=True)
    test_rdd = read_and_preprocess_data(spark, test_file, isTrain=False)
    
    # Check the number of rows of train and test data
    train_count = train_rdd.count()
    test_count = test_rdd.count()
    print(f"Train RDD count: {train_count}")
    print(f"Test RDD count: {test_count}")
    if train_count == 0:
        raise ValueError("Train RDD is empty after preprocessing")
    
    # Split train data into train and validation sets
    train_rdd, val_rdd = train_rdd.randomSplit([0.8, 0.2], seed=42)
    
    train_rdd = train_rdd.repartition(sc.defaultParallelism * 4).cache()
    val_rdd = val_rdd.repartition(sc.defaultParallelism * 4).cache()
    test_rdd = test_rdd.repartition(sc.defaultParallelism * 4)
    
    # Check the number of rows of validation data
    val_count = val_rdd.count()
    print(f"Validation RDD count: {val_count}")
    if val_count == 0:
        raise ValueError("Validation RDD is empty after split")
    
    # Grid search's combination
    model_candidate = {
        'maxDepths': list(range(1, 14)),
        'impurity': 'variance',
    }
    
    # Init best parameters
    best_model, best_rmse, best_depth, best_training_time, best_r2 = None, float('inf'), 0, 0, 0
    results = []
    
    # Loop through each combination in grid search
    for depth in model_candidate['maxDepths']:
        print(f"Training with depth: {depth}")
        
        # Train model and get training time
        train_start = time.time()
        model = DecisionTree.trainRegressor(
            train_rdd,
            categoricalFeaturesInfo={},
            maxDepth=depth,
            impurity=model_candidate["impurity"]
        )
        training_time = time.time() - train_start
        print(f"Training time: {training_time:.2f} seconds")
        
        # Evaluate on train and validation data
        train_rmse, train_r2 = evaluate_model(model, train_rdd, sc)
        val_rmse, val_r2 = evaluate_model(model, val_rdd, sc)
        print(f"Train RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}")
        
        # Get best model
        if val_rmse < best_rmse:
            best_rmse, best_r2, best_model = val_rmse, val_r2, model
            best_depth, best_training_time = depth, training_time
        
        results.append(f"Depth: {depth}, Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}, "
                       f"Validation RMSE: {val_rmse:.4f}, Validation R2: {val_r2:.4f}")
    
    # Predict on test data using best model
    print("Making predictions on test data...")
    test_features = test_rdd.map(lambda x: x[1])
    test_predictions = best_model.predict(test_features)
    test_ids = test_rdd.map(lambda x: x[0])
    test_predictions = test_ids.zip(test_predictions).map(
        lambda id_pred: (id_pred[0], math.exp(float(id_pred[1])) - 1)
    )
    
    # Create RDD from predicted data, ordered by id
    pred_df = spark.createDataFrame(test_predictions, ["id", "prediction"])
    test_predictions_df = pred_df.orderBy("id")
    
    # Get total processing time
    total_time = time.time() - start_time
    
    # Save best model and prediction results
    feature_cols = ["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude",
                    "dropoff_longitude", "dropoff_latitude", "day_of_month",
                    "month", "day_of_week", "hour", "store_and_fwd_flag", "peak_hour", "distance_haversine"]
    feature_importances = parse_feature_usage(best_model.toDebugString(), feature_cols)
    model_metrics = [
        ("Tree", best_model.toDebugString()),
        ("Estimated Feature Importances", str(feature_importances)),
        ("Valid RMSE", str(best_rmse)),
        ("Valid R2", str(best_r2)),
        ("Best Max Depth", str(best_depth)),
        ("Impurity", model_candidate["impurity"]),
        ("Training Time (Seconds)", str(best_training_time)),
        ("Total Processing Time (Seconds)", str(total_time))
    ]
    model_metrics_df = spark.createDataFrame(model_metrics, ["Metric", "Value"])
    model_metrics_df.coalesce(1).write.csv(output_file + "/model_results", header=True, mode="overwrite")
    test_predictions_df.coalesce(1).write.csv(output_file + "/predictions", header=True, mode="overwrite")
    
    print(f"Model results saved to {output_file}/model_results")
    print(f"Test predictions saved to {output_file}/predictions")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: TreeClassification <train_file> <test_file> <output_file>")
        sys.exit(-1)
    spark = SparkSession.builder \
        .appName("TreeClassification") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "4") \
        .config("spark.executor.instances", "4") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
        
    # Get parameters
    train_file, test_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    
    try:
        RDDTreeClassification(spark, train_file, test_file, output_file)
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down Spark...")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        spark.stop()