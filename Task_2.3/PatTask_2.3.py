from pyspark.sql import SparkSession
import sys
from math import sqrt, sin, cos, asin, radians, log1p, exp
import time
import datetime

def parse_train_line(line):
    """Parse and preprocess a train.csv line into (id, features, label) tuple."""
    cols = line.split(",")
    try:
        id_val = cols[0]
        vendor_id = float(cols[1])
        pickup_datetime = cols[2]
        dropoff_datetime = cols[3]
        passenger_count = float(cols[4])
        pickup_longitude = float(cols[5])
        pickup_latitude = float(cols[6])
        dropoff_longitude = float(cols[7])
        dropoff_latitude = float(cols[8])
        store_and_fwd_flag = 1.0 if cols[9] in ["Y", "yes"] else 0.0
        trip_duration = float(cols[10])

        def parse_datetime(datetime_str):
            try:
                dt = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                return {
                    "day_of_month": float(dt.day),
                    "month": float(dt.month),
                    "day_of_week": float(dt.weekday() + 1),
                    "hour": float(dt.hour),
                    "timestamp": dt.timestamp()
                }
            except:
                return None

        pickup_dt = parse_datetime(pickup_datetime)
        if pickup_dt is None:
            return None
        dropoff_dt = parse_datetime(dropoff_datetime)
        if dropoff_dt is None:
            return None

        calculated_duration = dropoff_dt["timestamp"] - pickup_dt["timestamp"]
        if abs(trip_duration - calculated_duration) > 120:
            return None

        if trip_duration <= 60 or trip_duration >= 6 * 3600:
            return None

        if not (-74.3 < pickup_longitude < -73.7 and 40.5 < pickup_latitude < 41.0 and
                -74.3 < dropoff_longitude < -73.7 and 40.5 < dropoff_latitude < 41.0):
            return None

        day_of_month = pickup_dt["day_of_month"]
        month = pickup_dt["month"]
        day_of_week = pickup_dt["day_of_week"]
        hour = pickup_dt["hour"]
        peak_hour = 1.0 if hour in (7, 8, 9, 16, 17, 18) else 0.0

        lat1_rad = radians(pickup_latitude)
        lon1_rad = radians(pickup_longitude)
        lat2_rad = radians(dropoff_latitude)
        lon2_rad = radians(dropoff_longitude)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        distance_haversine = c * 6371

        speed = distance_haversine / (trip_duration / 3600)
        if speed >= 100:
            return None

        features = [
            vendor_id, passenger_count, pickup_longitude, pickup_latitude,
            dropoff_longitude, dropoff_latitude, day_of_month, month,
            day_of_week, hour, store_and_fwd_flag, peak_hour, distance_haversine
        ]

        if any(v is None or (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf'))) for v in
               features):
            return None

        label = log1p(trip_duration)
        return (id_val, features, label)
    except:
        return None

def parse_test_line(line):
    """Parse and preprocess a test.csv line into (id, features) tuple."""
    cols = line.split(",")
    try:
        id_val = cols[0]
        vendor_id = float(cols[1])
        pickup_datetime = cols[2]
        passenger_count = float(cols[3])
        pickup_longitude = float(cols[4])
        pickup_latitude = float(cols[5])
        dropoff_longitude = float(cols[6])
        dropoff_latitude = float(cols[7])
        store_and_fwd_flag = 1.0 if cols[8] in ["Y", "yes"] else 0.0

        def parse_datetime(datetime_str):
            try:
                dt = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                return {
                    "day_of_month": float(dt.day),
                    "month": float(dt.month),
                    "day_of_week": float(dt.weekday() + 1),
                    "hour": float(dt.hour)
                }
            except:
                return None

        pickup_dt = parse_datetime(pickup_datetime)
        if pickup_dt is None:
            return None

        day_of_month = pickup_dt["day_of_month"]
        month = pickup_dt["month"]
        day_of_week = pickup_dt["day_of_week"]
        hour = pickup_dt["hour"]
        peak_hour = 1.0 if hour in (7, 8, 9, 16, 17, 18) else 0.0

        lat1_rad = radians(pickup_latitude)
        lon1_rad = radians(pickup_longitude)
        lat2_rad = radians(dropoff_latitude)
        lon2_rad = radians(dropoff_longitude)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        distance_haversine = c * 6371

        features = [
            vendor_id, passenger_count, pickup_longitude, pickup_latitude,
            dropoff_longitude, dropoff_latitude, day_of_month, month,
            day_of_week, hour, store_and_fwd_flag, peak_hour, distance_haversine
        ]

        if any(v is None or (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf'))) for v in
               features):
            return None

        return (id_val, features)
    except:
        return None

class LowLevelDecisionTreeRegressor:
    def __init__(self, max_depth, feature_names):
        """Initialize with max_depth and feature names passed as parameters."""
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.tree = None

    def compute_split(self, data, feature_idx, threshold):
        left = [x for x in data if x[1][feature_idx] <= threshold]
        right = [x for x in data if x[1][feature_idx] > threshold]
        return left, right

    def compute_variance(self, data):
        labels = [x[2] for x in data]
        if not labels:
            return 0.0, 0.0
        mean = sum(labels) / len(labels)
        variance = sum((label - mean) ** 2 for label in labels) / len(labels)
        return variance, len(labels)

    def find_best_split(self, data, feature_indices):
        best_variance_reduction = float("-inf")
        best_feature_idx = None
        best_threshold = None
        best_left = None
        best_right = None

        total_variance, total_count = self.compute_variance(data)
        if total_count == 0:
            return None, None, None, None

        for idx in feature_indices:
            values = sorted(set(x[1][idx] for x in data))
            if len(values) < 2:
                continue
            threshold = values[len(values) // 2]
            left, right = self.compute_split(data, idx, threshold)
            var_left, count_left = self.compute_variance(left)
            var_right, count_right = self.compute_variance(right)
            total_count_split = count_left + count_right
            if total_count_split == 0:
                continue
            weighted_variance = (var_left * count_left + var_right * count_right) / total_count_split
            variance_reduction = total_variance - weighted_variance
            if variance_reduction > best_variance_reduction:
                best_variance_reduction = variance_reduction
                best_feature_idx = idx
                best_threshold = threshold
                best_left = left
                best_right = right
        return best_feature_idx, best_threshold, best_left, best_right

    def build_tree(self, data, depth):
        if depth >= self.max_depth or len(data) < 2:
            labels = [x[2] for x in data]
            prediction = sum(labels) / len(labels) if labels else 0.0
            return {"leaf": True, "prediction": prediction}

        feature_indices = range(len(self.feature_names))
        feature_idx, threshold, left, right = self.find_best_split(data, feature_indices)
        if feature_idx is None:
            labels = [x[2] for x in data]
            prediction = sum(labels) / len(labels) if labels else 0.0
            return {"leaf": True, "prediction": prediction}

        return {
            "leaf": False,
            "feature_idx": feature_idx,
            "feature_name": self.feature_names[feature_idx],
            "threshold": threshold,
            "left": self.build_tree(left, depth + 1),
            "right": self.build_tree(right, depth + 1)
        }

    def fit(self, train_data):
        """Fit the decision tree to preprocessed training data."""
        if not train_data:
            raise ValueError("Training data is empty.")
        self.tree = self.build_tree(train_data, depth=0)
        return self

    def predict(self, features):
        """Predict trip_duration for a single feature vector."""
        if self.tree is None:
            raise ValueError("Tree has not been fitted.")
        node = self.tree
        while not node["leaf"]:
            if features[node["feature_idx"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"]

    def compute_feature_importances(self):
        feature_counts = {}

        def traverse_tree(node):
            if not node["leaf"]:
                feature_counts[node["feature_name"]] = feature_counts.get(node["feature_name"], 0) + 1
                traverse_tree(node["left"])
                traverse_tree(node["right"])

        if self.tree:
            traverse_tree(self.tree)
        total_splits = sum(feature_counts.values())
        if total_splits == 0:
            return {name: 0.0 for name in self.feature_names}

        importances = {
            name: (feature_counts.get(name, 0) / total_splits) if total_splits > 0 else 0.0
            for name in self.feature_names
        }
        return importances

    def tree_to_string(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if node is None:
            return "No tree fitted."
        indent = "  " * depth
        if node["leaf"]:
            return f"{indent}Predict: {node['prediction']:.2f}\n"
        else:
            return (f"{indent}If {node['feature_name']} <= {node['threshold']:.2f}\n" +
                    self.tree_to_string(node["left"], depth + 1) +
                    f"{indent}Else\n" +
                    self.tree_to_string(node["right"], depth + 1))

def load_and_preprocess_train_data(sc, file_path, seed=42):
    print("Creating RDD from train file...")
    lines = sc.textFile(file_path)
    print("Filtering header...")
    header = lines.first()
    data = lines.filter(lambda line: line != header).map(parse_train_line).filter(lambda x: x is not None)
    print("Train data RDD created.")
    print("Splitting data...")
    train_rdd, valid_rdd = data.randomSplit([0.8, 0.2], seed=seed)
    print("Caching train RDD...")
    train_rdd.cache()
    print("Caching valid RDD...")
    valid_rdd.cache()
    print("Counting train RDD...")
    train_count = train_rdd.count()
    print("Counting valid RDD...")
    valid_count = valid_rdd.count()
    print(f"Training set size: {train_count}, Validation set size: {valid_count}")
    return train_rdd, valid_rdd

def load_and_preprocess_test_data(sc, file_path):
    print("Creating RDD from test file...")
    lines = sc.textFile(file_path)
    print("Filtering header...")
    header = lines.first()
    data = lines.filter(lambda line: line != header).map(parse_test_line).filter(lambda x: x is not None)
    print("Test data RDD created.")
    print("Caching test RDD...")
    data.cache()
    print("Counting test RDD...")
    test_count = data.count()
    print(f"Test set size: {test_count}")
    return data

def grid_search(train_data, valid_rdd, feature_cols, max_depths):
    best_rmse = float("inf")
    best_r2 = float("-inf")
    best_model = None
    best_max_depth = None
    best_training_time = None

    print("Starting hyperparameter tuning...")
    for max_depth in max_depths:
        print(f"--- Training with maxDepth={max_depth} ---")
        dt = LowLevelDecisionTreeRegressor(max_depth=max_depth, feature_names=feature_cols)
        train_start_time = time.time()
        model = dt.fit(train_data)
        training_time = time.time() - train_start_time
        print(f"Training time for maxDepth={max_depth}: {training_time:.2f} seconds")

        # Evaluate on validation set
        valid_predictions = valid_rdd.map(lambda x: (x[2], model.predict(x[1])))
        valid_predictions = valid_predictions.map(lambda x: (exp(x[0]) - 1, exp(x[1]) - 1))  # Reverse log transform
        valid_mse = valid_predictions.map(lambda x: (x[0] - x[1]) ** 2).mean()
        valid_rmse = sqrt(valid_mse)

        # Calculate R²
        valid_labels = valid_predictions.map(lambda x: x[0]).collect()
        mean_label = sum(valid_labels) / len(valid_labels)
        ss_tot = sum((y - mean_label) ** 2 for y in valid_labels)
        ss_res = valid_mse * len(valid_labels)
        valid_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        print(f"RMSE (validation, original scale) for maxDepth={max_depth}: {valid_rmse}")
        print(f"R² (validation, original scale) for maxDepth={max_depth}: {valid_r2}")

        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_r2 = valid_r2
            best_model = model
            best_max_depth = max_depth
            best_training_time = training_time

    return best_model, best_rmse, best_r2, best_max_depth, best_training_time

def main():
    if len(sys.argv) != 4:
        print("Usage: spark-submit script.py <train_input_path> <test_input_path> <output_path>")
        sys.exit(1)

    train_input_path = sys.argv[1]
    test_input_path = sys.argv[2]
    output_path = sys.argv[3]

    total_start_time = time.time()
    print("Initializing SparkSession...")
    spark = SparkSession.builder.appName("LowLevelDecisionTree").getOrCreate()
    sc = spark.sparkContext

    feature_cols = [
        "vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "day_of_month",
        "month", "day_of_week", "hour", "store_and_fwd_flag",
        "peak_hour", "distance_haversine"
    ]

    try:
        # Load and preprocess data
        print("Loading and preprocessing training data...")
        train_rdd, valid_rdd = load_and_preprocess_train_data(sc, train_input_path)
        print("Collecting training data...")
        train_data = train_rdd.collect()

        # Grid search for best maxDepth
        max_depths = list(range(1, 15))  # 1 to 14
        best_model, best_rmse, best_r2, best_max_depth, best_training_time = grid_search(
            train_data, valid_rdd, feature_cols, max_depths
        )
        print(f"\nBest model: maxDepth={best_max_depth}, impurity=variance")
        print(f"Best RMSE: {best_rmse}")
        print(f"Best R²: {best_r2}")
        print(f"Training time for best model: {best_training_time:.2f} seconds")

        # Evaluate on training set
        print("Computing train RMSE...")
        train_predictions = train_rdd.map(lambda x: (x[2], best_model.predict(x[1])))
        train_predictions = train_predictions.map(lambda x: (exp(x[0]) - 1, exp(x[1]) - 1))
        train_mse = train_predictions.map(lambda x: (x[0] - x[1]) ** 2).mean()
        train_rmse = sqrt(train_mse)
        print(f"RMSE on training set: {train_rmse}")

        # Compute feature importances and tree structure
        print("Computing feature importances...")
        feature_importances = best_model.compute_feature_importances()
        print("Generating tree structure...")
        tree_structure = best_model.tree_to_string()

        # Predict on test data
        print("Loading and preprocessing test data...")
        test_rdd = load_and_preprocess_test_data(sc, test_input_path)
        print("Generating predictions...")
        predictions_rdd = test_rdd.map(lambda x: (x[0], exp(best_model.predict(x[1])) - 1))  # Reverse log transform

        print("Generating sample predictions...")
        sample_data = test_rdd.map(lambda x: (x[0], x[1])).take(5)
        sample_predictions = []
        for id_val, features in sample_data:
            pred = exp(best_model.predict(features)) - 1
            sample_predictions.append((id_val, dict(zip(feature_cols, features)), pred))
            print(f"ID: {id_val}, Features: {dict(zip(feature_cols, features))}, Predicted: {pred}")

        print("Saving sample predictions...")
        sample_df = spark.createDataFrame(
            [(id_val, str(features), predicted) for id_val, features, predicted in sample_predictions],
            ["ID", "Features", "Predicted"]
        )
        sample_df.coalesce(1).write.mode("overwrite").csv(f"{output_path}/sample_predictions", header=True)
        print(f"Sample predictions saved to: {output_path}/sample_predictions")

        print("Saving all predictions...")
        predictions_df = spark.createDataFrame(predictions_rdd, ["id", "prediction"]).orderBy("id")
        predictions_df.coalesce(1).write.mode("overwrite").csv(f"{output_path}/predictions", header=True)
        print(f"All predictions saved to: {output_path}/predictions")

        # Calculate total processing time
        total_processing_time = time.time() - total_start_time

        # Save model metrics
        print("Preparing model metrics...")
        model_metrics = [
            ("Tree", tree_structure),
            ("Features Importance", str(feature_importances)),
            ("Valid RMSE", str(best_rmse)),
            ("Valid R2", str(best_r2)),
            ("Best Max Depth", str(best_max_depth)),
            ("Impurity", "variance"),  # Fixed for regression
            ("Training Time (Seconds)", str(best_training_time)),
            ("Total Processing Time (Seconds)", str(total_processing_time)),
        ]
        print("Saving model metrics...")
        model_metrics_df = spark.createDataFrame(model_metrics, ["Metric", "Value"])
        model_metrics_df.coalesce(1).write.mode("overwrite").csv(f"{output_path}/model_metrics", header=True)
        print(f"Model metrics saved to: {output_path}/model_metrics")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Stopping SparkSession...")
        spark.stop()

if __name__ == "__main__":
    main()
