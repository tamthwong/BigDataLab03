from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, lit, unix_timestamp, col, when, dayofmonth, month, hour, dayofweek, monotonically_increasing_id, abs
from pyspark.sql.types import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import Vectors
import sys
import time

# Evaluate the model using RMSE and R2 with partition-based processing
def evaluate_model(model, rdd, sc, num_partitions=100):
    print("Evaluating model...")
    
    # Chia RDD thành số lượng partitions mong muốn
    partitioned_rdd = rdd.repartition(num_partitions)
    total_partitions = partitioned_rdd.getNumPartitions()
    print(f"Total partitions: {total_partitions}")
    
    # Hàm xử lý từng partition
    def process_partition(iterator):
        # Thu thập dữ liệu từ partition hiện tại
        partition_data = [(lp.features.toArray(), lp.label) for lp in iterator]
        if not partition_data:
            return []
        # Dự đoán trên partition
        partition_predictions = [float(model.predict(features)) for features, _ in partition_data]
        partition_labels = [label for _, label in partition_data]
        return [(p, l) for p, l in zip(partition_predictions, partition_labels)]
    
    # Áp dụng hàm xử lý trên từng partition và thu thập kết quả
    print("Processing partitions...")
    start_time = time.time()
    predictions_labels_rdd = partitioned_rdd.mapPartitions(process_partition)
    all_predictions_labels = predictions_labels_rdd.collect()
    
    # Tách predictions và labels
    all_predictions = [p for p, _ in all_predictions_labels]
    all_labels = [l for _, l in all_predictions_labels]
    
    print(f"Total records processed: {len(all_predictions)}")
    print(f"Partition processing time: {(time.time() - start_time):.2f} seconds")
    print("Calculating RMSE and R2...")
    
    # Tạo RDD từ kết quả dự đoán và nhãn
    predictions_and_labels = sc.parallelize(list(zip(all_predictions, all_labels)))
    metrics = RegressionMetrics(predictions_and_labels)
    rmse = metrics.rootMeanSquaredError
    r2 = metrics.r2
    return rmse, r2

# Read and preprocess data
def read_and_preprocess_data(spark, input_file, is_train=True):
    df = spark.read.csv(input_file, header=True, inferSchema=True)
    df = df.na.drop()
    
    df = df.withColumn("pickup_datetime", to_timestamp("pickup_datetime"))
    
    if is_train:
        df = df.withColumn("dropoff_datetime", to_timestamp("dropoff_datetime"))
        df = df.withColumn("actual_duration", unix_timestamp("dropoff_datetime") - unix_timestamp("pickup_datetime"))
        df = df.filter(abs(col("actual_duration") - col("trip_duration")) <= lit(120))
        df = df.drop("actual_duration", "dropoff_datetime")
    else:
        df = df.withColumn("row_index", monotonically_increasing_id())
    
    df = df.withColumn("flag_binary", when(col("store_and_fwd_flag") == lit("Y"), lit(1)).otherwise(lit(0)))
    df = df.withColumn("day_of_month", dayofmonth("pickup_datetime"))
    df = df.withColumn("month", month("pickup_datetime"))
    df = df.withColumn("day_of_week", when(dayofweek("pickup_datetime") == lit(1), lit(7)).otherwise(dayofweek("pickup_datetime") - lit(1)))
    df = df.withColumn("hour", hour("pickup_datetime"))
    
    feature_cols = ["vendor_id", "flag_binary", "day_of_month", "month", "day_of_week", "hour",
                    "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    
    if is_train:
        df = df.select(feature_cols + ["trip_duration"])
        rdd = df.rdd.map(lambda row: LabeledPoint(row["trip_duration"], Vectors.dense([row[c] for c in feature_cols])))
        return rdd, None
    else:
        df_features = df.select(feature_cols)
        rdd = df_features.rdd.map(lambda row: Vectors.dense([row[c] for c in feature_cols]))
        id_df = df.select("id", "row_index")
        return rdd, id_df

# Decision Tree Regression
def RDDTreeClassification(spark, train_file, test_file, output_file):
    sc = spark.sparkContext
    
    results = []

    # Read and preprocess data
    train_rdd, _ = read_and_preprocess_data(spark, train_file, is_train=True)
    test_rdd, test_id_df = read_and_preprocess_data(spark, test_file, is_train=False)
    
    # Split train data into train and validation sets
    train_rdd, val_rdd = train_rdd.randomSplit([0.8, 0.2], seed=42)
    
    # Cache RDD để tăng tốc độ
    train_rdd.cache()
    val_rdd.cache()
    
    # In số lượng bản ghi để kiểm tra
    print(f"Number of records in train_rdd: {train_rdd.count()}")
    print(f"Number of records in val_rdd: {val_rdd.count()}")
    
    # Grid search
    model_candidate = {
        'maxDepths': [5, 10, 15],
        'impurity': ['variance'],
        'maxBins': [32, 64, 128]
    }
    
    best_model = None
    best_rmse = float('inf')
    best_depth = 0
    best_bins = 0
    
    for depth in model_candidate['maxDepths']:
        for bins in model_candidate['maxBins']:
            print(f"-------------------->>Depth: {depth}, Bins: {bins}")
            
            start_time = time.time()
            model = DecisionTree.trainRegressor(train_rdd, categoricalFeaturesInfo={}, maxDepth=depth, maxBins=bins, impurity=model_candidate["impurity"][0])
            end_time = time.time()
            print(f"----------{depth}, {bins}----------\nTraining time: {(end_time - start_time):.2f} seconds")
            
            train_rmse, train_r2 = evaluate_model(model, train_rdd, sc)
            print(f"----------\nTrain RMSE: {train_rmse}\nR2: {train_r2}")
            
            val_rmse, val_r2 = evaluate_model(model, val_rdd, sc)
            print(f"----------\nValidation RMSE: {val_rmse}\nR2: {val_r2}")
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_model = model
                best_depth = depth
                best_bins = bins
                
            results.append(f"Depth: {depth}, Bins: {bins}, Train RMSE: {train_rmse}, Train R2: {train_r2}, Validation RMSE: {val_rmse}, Validation R2: {val_r2}")
    
    # Predict on test data
    test_features = test_rdd.collect()
    test_predictions = [float(best_model.predict(features)) for features in test_features]
    test_predictions_rdd = sc.parallelize(list(enumerate(test_predictions))).map(lambda x: (x[0], x[1]))
    
    # Convert predictions RDD to DataFrame
    pred_df = spark.createDataFrame(test_predictions_rdd, ["row_index", "prediction"])
    
    # Join predictions with test id DataFrame
    result_df = test_id_df.join(pred_df, "row_index").select("id", "prediction")
    
    # Save model results
    tree_structure = best_model.toDebugString()
    results.append(f"Best parameters: Depth: {best_depth}, Bins: {best_bins}, Validation RMSE: {best_rmse}")
    results.append(f"Tree Structure:\n{tree_structure}")
    sc.parallelize(results).coalesce(1).saveAsTextFile(output_file + "/model_results")
    
    # Save predictions
    result_df.write.csv(output_file + "/predictions", header=True, mode="overwrite")
    
    # Unpersist cached RDD
    train_rdd.unpersist()
    val_rdd.unpersist()
    
    print(f"Tree Structure saved to {output_file}/model_results")
    print(f"Test predictions (id, prediction) saved to {output_file}/predictions")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: TreeClassification <train_file> <test_file> <output_file>")
        sys.exit(-1)
        
    spark = SparkSession.builder.appName("TreeClassification").getOrCreate()

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    RDDTreeClassification(spark, train_file, test_file, output_file)

    spark.stop()