from pyspark.sql import SparkSession
from math import sqrt, sin, cos, asin, radians, log1p, exp
import sys
import time
import datetime


def parse_train_line(line):
    """
    Parse a single line from train.csv into (id, features, label) with preprocessing.

    Args:
        line: String representing a CSV row

    Returns:
        Tuple of (id, features, label) or None if parsing fails
    """
    cols = line.split(",")
    try:
        id_val = cols[0]
        vendor_id = float(cols[1])

        # Parse pickup and dropoff datetimes
        pickup_dt = parse_datetime(cols[2])
        dropoff_dt = parse_datetime(cols[3])
        if not (pickup_dt and dropoff_dt):
            return None

        passenger_count = float(cols[4])
        pickup_longitude, pickup_latitude = float(cols[5]), float(cols[6])
        dropoff_longitude, dropoff_latitude = float(cols[7]), float(cols[8])

        # Encode store_and_fwd_flag as binary (1 for 'Y'/'yes', 0 otherwise)
        store_and_fwd_flag = 1.0 if cols[9] in ["Y", "yes"] else 0.0
        trip_duration = float(cols[10])

        # Verify trip duration consistency (within 2 minutes of calculated duration)
        calculated_duration = dropoff_dt["timestamp"] - pickup_dt["timestamp"]
        if abs(trip_duration - calculated_duration) > 120:
            return None

        # Filter unrealistic trip durations (< 60s or > 6h)
        if trip_duration <= 60 or trip_duration >= 6 * 3600:
            return None

        # Restrict coordinates to NYC bounds
        if not (
                -74.3 < pickup_longitude < -73.7 and 40.5 < pickup_latitude < 41.0 and
                -74.3 < dropoff_longitude < -73.7 and 40.5 < dropoff_latitude < 41.0
        ):
            return None

        # Extract temporal features
        day_of_month, month = pickup_dt["day_of_month"], pickup_dt["month"]
        day_of_week, hour = pickup_dt["day_of_week"], pickup_dt["hour"]

        # Flag peak hours (7-9 AM, 4-6 PM)
        peak_hour = 1.0 if hour in (7, 8, 9, 16, 17, 18) else 0.0

        # Compute Haversine distance
        lat1, lon1 = radians(pickup_latitude), radians(pickup_longitude)
        lat2, lon2 = radians(dropoff_latitude), radians(dropoff_longitude)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        distance_haversine = 2 * asin(sqrt(a)) * 6371

        # Filter trips with speed > 100 km/h
        speed = distance_haversine / (trip_duration / 3600)
        if speed >= 100:
            return None

        # Assemble feature list
        features = [
            vendor_id, passenger_count, pickup_longitude, pickup_latitude,
            dropoff_longitude, dropoff_latitude, day_of_month, month,
            day_of_week, hour, store_and_fwd_flag, peak_hour, distance_haversine
        ]

        # Validate features for invalid values
        if any(
                v is None or
                (isinstance(v, float) and (v != v or v in (float('inf'), float('-inf'))))
                for v in features
        ):
            return None

        # Return processed data with log-transformed label
        return (id_val, features, log1p(trip_duration))
    except:
        return None


def parse_test_line(line):
    """
    Parse a single line from test.csv into (id, features) with preprocessing.

    Args:
        line: String representing a CSV row

    Returns:
        Tuple of (id, features) or None if parsing fails
    """
    cols = line.split(",")
    try:
        id_val = cols[0]
        vendor_id = float(cols[1])

        # Parse pickup datetime
        pickup_dt = parse_datetime(cols[2])
        if not pickup_dt:
            return None

        passenger_count = float(cols[3])
        pickup_longitude, pickup_latitude = float(cols[4]), float(cols[5])
        dropoff_longitude, dropoff_latitude = float(cols[6]), float(cols[7])

        # Encode store_and_fwd_flag
        store_and_fwd_flag = 1.0 if cols[8] in ["Y", "yes"] else 0.0

        # Extract temporal features
        day_of_month, month = pickup_dt["day_of_month"], pickup_dt["month"]
        day_of_week, hour = pickup_dt["day_of_week"], pickup_dt["hour"]
        peak_hour = 1.0 if hour in (7, 8, 9, 16, 17, 18) else 0.0

        # Compute Haversine distance
        lat1, lon1 = radians(pickup_latitude), radians(pickup_longitude)
        lat2, lon2 = radians(dropoff_latitude), radians(dropoff_longitude)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        distance_haversine = 2 * asin(sqrt(a)) * 6371

        # Assemble feature list
        features = [
            vendor_id, passenger_count, pickup_longitude, pickup_latitude,
            dropoff_longitude, dropoff_latitude, day_of_month, month,
            day_of_week, hour, store_and_fwd_flag, peak_hour, distance_haversine
        ]

        # Validate features
        if any(
                v is None or
                (isinstance(v, float) and (v != v or v in (float('inf'), float('-inf'))))
                for v in features
        ):
            return None

        return (id_val, features)
    except:
        return None


def parse_datetime(datetime_str):
    """
    Convert datetime string to a dictionary of features.

    Args:
        datetime_str: String in format 'YYYY-MM-DD HH:MM:SS'

    Returns:
        Dictionary with day_of_month, month, day_of_week, hour, and timestamp, or None if parsing fails
    """
    try:
        dt = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        return {
            "day_of_month": float(dt.day),
            "month": float(dt.month),
            "day_of_week": float(dt.weekday() + 1),  # Monday=1, Sunday=7
            "hour": float(dt.hour),
            "timestamp": dt.timestamp()
        }
    except:
        return None


class LowLevelDecisionTreeRegressor:
    """Custom decision tree regressor for predicting continuous values."""

    def __init__(self, max_depth, feature_names):
        """
        Initialize the decision tree.

        Args:
            max_depth: Maximum depth of the tree
            feature_names: List of feature names
        """
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.tree = None

    def compute_split(self, data, feature_idx, threshold):
        """
        Split data based on a feature and threshold.

        Args:
            data: List of (id, features, label) tuples
            feature_idx: Index of the feature to split on
            threshold: Threshold value for the split

        Returns:
            Tuple of (left, right) data subsets
        """
        left = [x for x in data if x[1][feature_idx] <= threshold]
        right = [x for x in data if x[1][feature_idx] > threshold]
        return left, right

    def compute_variance(self, data):
        """
        Calculate variance and count of labels in the data.

        Args:
            data: List of (id, features, label) tuples

        Returns:
            Tuple of (variance, count)
        """
        labels = [x[2] for x in data]
        if not labels:
            return 0.0, 0.0
        mean = sum(labels) / len(labels)
        variance = sum((label - mean) ** 2 for label in labels) / len(labels)
        return variance, len(labels)

    def find_best_split(self, data, feature_indices):
        """
        Find the best feature and threshold for splitting data.

        Args:
            data: List of (id, features, label) tuples
            feature_indices: Indices of features to consider

        Returns:
            Tuple of (best_feature_idx, best_threshold, best_left, best_right)
        """
        best_variance_reduction = float("-inf")
        best_feature_idx, best_threshold, best_left, best_right = None, None, None, None
        total_variance, total_count = self.compute_variance(data)
        if total_count == 0:
            return None, None, None, None

        for idx in feature_indices:
            values = sorted(set(x[1][idx] for x in data))
            if len(values) < 2:
                continue
            threshold = values[len(values) // 2]  # Use median as threshold
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
                best_feature_idx, best_threshold = idx, threshold
                best_left, best_right = left, right
        return best_feature_idx, best_threshold, best_left, best_right

    def build_tree(self, data, depth):
        """
        Recursively build the decision tree.

        Args:
            data: List of (id, features, label) tuples
            depth: Current depth in the tree

        Returns:
            Dictionary representing a tree node
        """
        if depth >= self.max_depth or len(data) < 2:
            labels = [x[2] for x in data]
            return {"leaf": True, "prediction": sum(labels) / len(labels) if labels else 0.0}

        feature_indices = range(len(self.feature_names))
        feature_idx, threshold, left, right = self.find_best_split(data, feature_indices)
        if feature_idx is None:
            labels = [x[2] for x in data]
            return {"leaf": True, "prediction": sum(labels) / len(labels) if labels else 0.0}

        return {
            "leaf": False,
            "feature_idx": feature_idx,
            "feature_name": self.feature_names[feature_idx],
            "threshold": threshold,
            "left": self.build_tree(left, depth + 1),
            "right": self.build_tree(right, depth + 1)
        }

    def fit(self, train_data):
        """
        Train the decision tree on the provided data.

        Args:
            train_data: List of (id, features, label) tuples

        Returns:
            Self (trained model)
        """
        if not train_data:
            raise ValueError("Training data is empty.")
        self.tree = self.build_tree(train_data, depth=0)
        return self

    def predict(self, features):
        """
        Predict the label for a single feature vector.

        Args:
            features: List of feature values

        Returns:
            Predicted label (log scale)
        """
        if self.tree is None:
            raise ValueError("Tree has not been fitted.")
        node = self.tree
        while not node["leaf"]:
            node = node["left"] if features[node["feature_idx"]] <= node["threshold"] else node["right"]
        return node["prediction"]

    def trace_prediction(self, features, feature_names):
        """
        Trace the decision path for a feature vector.

        Args:
            features: List of feature values
            feature_names: List of feature names

        Returns:
            List of strings describing the decision path
        """
        if self.tree is None:
            raise ValueError("Tree has not been fitted.")
        path = []
        node = self.tree
        while not node["leaf"]:
            feature_idx = node["feature_idx"]
            feature_name = node["feature_name"]
            threshold = node["threshold"]
            feature_value = features[feature_idx]
            if feature_value <= threshold:
                path.append(f"{feature_name} = {feature_value:.4f} <= {threshold:.4f}, go left")
                node = node["left"]
            else:
                path.append(f"{feature_name} = {feature_value:.4f} > {threshold:.4f}, go right")
                node = node["right"]
        path.append(
            f"Reached leaf, predict: {node['prediction']:.4f} "
            f"(original scale: {exp(node['prediction']) - 1:.2f} seconds)"
        )
        return path

    def compute_feature_importances(self):
        """
        Calculate feature importance based on frequency of splits.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        feature_counts = {}

        def traverse_tree(node):
            if not node["leaf"]:
                feature_counts[node["feature_name"]] = feature_counts.get(node["feature_name"], 0) + 1
                traverse_tree(node["left"])
                traverse_tree(node["right"])

        if self.tree:
            traverse_tree(self.tree)
        total_splits = sum(feature_counts.values())
        return {
            name: feature_counts.get(name, 0) / total_splits if total_splits > 0 else 0.0
            for name in self.feature_names
        }

    def tree_to_string(self, node=None, depth=0):
        """
        Convert the tree to a readable string.

        Args:
            node: Current tree node (default: root)
            depth: Current depth for indentation

        Returns:
            String representation of the tree
        """
        node = node or self.tree
        if not node:
            return "No tree fitted."
        indent = "  " * depth
        if node["leaf"]:
            return f"{indent}Predict: {node['prediction']:.2f}\n"
        return (
                f"{indent}If {node['feature_name']} <= {node['threshold']:.2f}\n" +
                self.tree_to_string(node["left"], depth + 1) +
                f"{indent}Else\n" +
                self.tree_to_string(node["right"], depth + 1)
        )


def load_and_preprocess_train_data(sc, file_path, seed=42):
    """
    Load and split training data into training and validation RDDs.

    Args:
        sc: SparkContext
        file_path: Path to training CSV
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_rdd, valid_rdd)
    """
    lines = sc.textFile(file_path)
    header = lines.first()
    data = lines.filter(lambda line: line != header) \
                .map(parse_train_line) \
                .filter(lambda x: x is not None)
    train_rdd, valid_rdd = data.randomSplit([0.8, 0.2], seed=seed)
    train_rdd.cache()
    valid_rdd.cache()
    print(f"Training set size: {train_rdd.count()}, Validation set size: {valid_rdd.count()}")
    return train_rdd, valid_rdd


def load_and_preprocess_test_data(sc, file_path):
    """
    Load and preprocess test data into an RDD.

    Args:
        sc: SparkContext
        file_path: Path to test CSV

    Returns:
        RDD of (id, features) tuples
    """
    lines = sc.textFile(file_path)
    header = lines.first()
    data = lines.filter(lambda line: line != header) \
                .map(parse_test_line) \
                .filter(lambda x: x is not None)
    data.cache()
    print(f"Test set size: {data.count()}")
    return data


def grid_search(train_data, valid_rdd, feature_cols, max_depths):
    """
    Perform grid search over max_depth to find the best model.

    Args:
        train_data: List of (id, features, label) tuples
        valid_rdd: Validation RDD
        feature_cols: List of feature names
        max_depths: List of max_depth values to try

    Returns:
        Tuple of (best_model, best_rmse, best_r2, best_max_depth, best_training_time)
    """
    best_rmse, best_r2 = float("inf"), float("-inf")
    best_model, best_max_depth, best_training_time = None, None, None

    print("Starting hyperparameter tuning...")
    for max_depth in max_depths:
        print(f"--- Training with maxDepth={max_depth} ---")
        dt = LowLevelDecisionTreeRegressor(max_depth, feature_cols)

        # Time the training process
        start_time = time.time()
        dt.fit(train_data)
        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")

        # Generate predictions on validation set
        valid_predictions = valid_rdd.map(
            lambda x: (exp(x[2]) - 1, exp(dt.predict(x[1])) - 1)
        )

        # Compute RMSE
        valid_mse = valid_predictions.map(lambda x: (x[0] - x[1]) ** 2).mean()
        valid_rmse = sqrt(valid_mse)

        # Compute R²
        valid_labels = valid_predictions.map(lambda x: x[0]).collect()
        mean_label = sum(valid_labels) / len(valid_labels)
        ss_tot = sum((y - mean_label) ** 2 for y in valid_labels)
        ss_res = valid_mse * len(valid_labels)
        valid_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        print(f"RMSE (validation): {valid_rmse:.2f}, R²: {valid_r2:.4f}")
        if valid_rmse < best_rmse:
            best_rmse, best_r2 = valid_rmse, valid_r2
            best_model, best_max_depth, best_training_time = dt, max_depth, training_time

    return best_model, best_rmse, best_r2, best_max_depth, best_training_time


def main():
    """
    Entry point for the script, handling input arguments and execution.

    Expects command-line arguments: train_input_path, test_input_path, output_path
    """
    if len(sys.argv) != 4:
        print("Usage: spark-submit script.py <train_input_path> <test_input_path> <output_path>")
        sys.exit(1)

    train_input_path, test_input_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # Start timing the entire process
    total_start_time = time.time()

    # Initialize Spark session
    spark = SparkSession.builder.appName("LowLevelDecisionTree").getOrCreate()
    sc = spark.sparkContext

    # Define feature columns
    feature_cols = [
        "vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "day_of_month", "month",
        "day_of_week", "hour", "store_and_fwd_flag", "peak_hour", "distance_haversine"
    ]

    try:
        # Load and preprocess training data
        train_rdd, valid_rdd = load_and_preprocess_train_data(sc, train_input_path)
        train_data = train_rdd.collect()

        # Perform grid search to find best model
        max_depths = list(range(1, 21))
        best_model, best_rmse, best_r2, best_max_depth, best_training_time = grid_search(
            train_data, valid_rdd, feature_cols, max_depths
        )
        print(
            f"\nBest model: maxDepth={best_max_depth}, "
            f"RMSE={best_rmse:.2f}, R²={best_r2:.4f}, Training time={best_training_time:.2f}s"
        )

        # Compute training RMSE
        train_predictions = train_rdd.map(
            lambda x: (exp(x[2]) - 1, exp(best_model.predict(x[1])) - 1)
        )
        train_rmse = sqrt(train_predictions.map(lambda x: (x[0] - x[1]) ** 2).mean())
        print(f"Train RMSE: {train_rmse:.2f}")

        # Extract feature importances and tree structure
        feature_importances = best_model.compute_feature_importances()
        tree_structure = best_model.tree_to_string()

        # Load and preprocess test data
        test_rdd = load_and_preprocess_test_data(sc, test_input_path)

        # Generate predictions
        predictions_rdd = test_rdd.map(
            lambda x: (x[0], exp(best_model.predict(x[1])) - 1)
        )
        predictions = predictions_rdd.collect()

        # Save predictions to CSV
        formatted_predictions = [f"{id_val},{pred:.2f}" for id_val, pred in predictions]
        formatted_predictions.insert(0, "id,prediction")
        sc.parallelize(formatted_predictions) \
          .coalesce(1) \
          .saveAsTextFile(f"{output_path}/predictions")
        print(f"Predictions saved to: {output_path}/predictions")

        # Calculate total processing time
        total_processing_time = time.time() - total_start_time

        # Save model metrics
        model_metrics = [
            f"Tree Structure: {tree_structure}",
            f"Feature Importances: {feature_importances}",
            f"Validation RMSE: {best_rmse:.2f}",
            f"Validation R2: {best_r2:.4f}",
            f"Best Max Depth: {best_max_depth}",
            f"Impurity: variance",
            f"Training Time (Seconds): {best_training_time:.2f}",
            f"Total Processing Time (Seconds): {total_processing_time:.2f}"
        ]
        sc.parallelize(model_metrics) \
          .coalesce(1) \
          .saveAsTextFile(f"{output_path}/model_metrics")
        print(f"Model metrics saved to: {output_path}/model_metrics")

        # Print performance summary
        print("\nSummary of Performance Metrics:")
        print(f"Training Time: {best_training_time:.2f} seconds")
        print(f"Total Processing Time: {total_processing_time:.2f} seconds")

        # Display sample predictions with decision paths
        print("\nSample Predictions (first 3 test records):")
        sample_data = test_rdd.take(3)
        for id_val, features in sample_data:
            feature_dict = dict(zip(feature_cols, features))
            path = best_model.trace_prediction(features, feature_cols)
            print(f"\nID: {id_val}")
            print("Features:", {k: f"{v:.4f}" for k, v in feature_dict.items()})
            print("Tree Traversal:")
            for step in path:
                print(f"  {step}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure Spark session is closed
        spark.stop()


if __name__ == "__main__":
    main()
