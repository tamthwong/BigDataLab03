import sys
from pyspark.sql import SparkSession
import math
from itertools import product
import random
import time

def z_score_normalize(rdd, mean_values, std_values):
    """
    Normalize features using z-score normalization (subtract mean, divide by standard deviation).
    
    Args:
        rdd: RDD of (features, label) tuples, where features include a bias term
        mean_values: List of mean values for each feature
        std_values: List of standard deviations for each feature
    
    Returns:
        RDD with normalized features and unchanged labels
    """
    return rdd.map(
        lambda row: (
            tuple(
                [(x - mean) / std if std != 0 else 0 for x, mean, std in zip(row[0], mean_values, std_values)]
                + [1.0]  # Preserve bias term
            ),
            row[1]
        )
    )

def compute_mean_std(rdd):
    """
    Compute mean and standard deviation for each feature in the RDD.
    
    Args:
        rdd: RDD of (features, label) tuples
    
    Returns:
        Tuple of (mean_vector, std_vector) for feature normalization
    """
    count = rdd.count()
    
    # Calculate sum and sum of squares in a single pass
    def combine_stats(a, b):
        return (
            [x + y for x, y in zip(a[0], b[0])],  # Sum of features
            [x + y for x, y in zip(a[1], b[1])]   # Sum of squared features
        )
    
    sum_vector, sum_squares = rdd.map(
        lambda x: ([xi for xi in x[0]], [xi * xi for xi in x[0]])
    ).reduce(combine_stats)
    
    # Compute mean for each feature
    mean_vector = [x / count for x in sum_vector]
    
    # Compute standard deviation with small constant to avoid division by zero
    std_vector = [
        math.sqrt((s / count) - (m * m) + 1e-10)
        for s, m in zip(sum_squares, mean_vector)
    ]
    return mean_vector, std_vector

def parse_line(line):
    """
    Parse a single CSV line into features and label.
    
    Args:
        line: String representing a CSV row
    
    Returns:
        Tuple of (features, label) where features are floats and label is an integer
    """
    parts = line.split(",")
    features = tuple(map(float, parts[1:-1]))  # Extract features (excluding id and label)
    raw_label = parts[-1].replace('"', '')     # Clean label
    label = int(raw_label)
    return (features, label)

def load_data(input_path, sc):
    """
    Load CSV data into an RDD, skipping the header.
    
    Args:
        input_path: Path to the input CSV file
        sc: SparkContext
    
    Returns:
        RDD of (features, label) tuples
    """
    lines = sc.textFile(input_path)
    header = lines.first()
    data = lines.filter(lambda line: line != header)  # Exclude header
    rdd_data = data.map(parse_line)
    return rdd_data

def preprocessing(rdd_data):
    """
    Preprocess data by removing duplicates, filtering invalid entries, splitting data,
    and normalizing features.
    
    Args:
        rdd_data: RDD of (features, label) tuples
    
    Returns:
        Tuple of (normalized_train_rdd, normalized_val_rdd, normalized_test_rdd)
    """
    # Remove duplicate rows
    rdd_data = rdd_data.distinct()
    
    # Filter out rows with NaN values
    rdd_data = rdd_data.filter(
        lambda x: all(not math.isnan(f) for f in x[0]) and not math.isnan(x[1])
    )
    
    # Split data into training (70%), validation (10%), and test (20%) sets
    train_rdd, val_rdd, test_rdd = rdd_data.randomSplit([0.7, 0.1, 0.2], seed=42)
    
    # Compute mean and std for training data
    mean_values, std_values = compute_mean_std(train_rdd)
    
    # Normalize features using z-score normalization
    normalized_train_rdd = z_score_normalize(train_rdd, mean_values, std_values).cache()
    normalized_val_rdd = z_score_normalize(val_rdd, mean_values, std_values).cache()
    normalized_test_rdd = z_score_normalize(test_rdd, mean_values, std_values).cache()
    
    return normalized_train_rdd, normalized_val_rdd, normalized_test_rdd

def dot_product(weights, features):
    """
    Compute the dot product of weights and features.
    
    Args:
        weights: List of model weights (including bias)
        features: List of feature values (including bias term)
    
    Returns:
        Float representing the dot product
    """
    return sum(w * f for w, f in zip(weights, features))

def sigmoid(z):
    """
    Apply the sigmoid function to a value, with clipping for stability.
    
    Args:
        z: Input value
    
    Returns:
        Sigmoid output between 0 and 1
    """
    if z > 100:
        return 1.0
    elif z < -100:
        return 0.0
    return 1 / (1 + math.exp(-z))

def compute_gradient(point, weights, reg_param):
    """
    Compute the gradient for a single data point.
    
    Args:
        point: Tuple of (features, label)
        weights: List of model weights
        reg_param: Regularization parameter for L2 penalty
    
    Returns:
        List of gradient values for each weight
    """
    features, label = point
    pred = sigmoid(dot_product(weights, features))
    grad = [f * (pred - label) for f in features]  # Gradient for logistic loss
    # Apply L2 regularization to all weights except bias
    return [
        g + reg_param * w if i < len(weights) - 1 else g
        for i, (g, w) in enumerate(zip(grad, weights))
    ]

def compute_loss(rdd, weights, reg_param, class_weights):
    """
    Compute the logistic loss with L2 regularization.
    
    Args:
        rdd: RDD of (features, label) tuples
        weights: List of model weights
        reg_param: Regularization parameter
        class_weights: Dictionary mapping labels to weights
    
    Returns:
        Total loss (log loss + L2 penalty)
    """
    temp_rdd = rdd.map(
        lambda p: class_weights[p[1]] * (
            -p[1] * math.log(sigmoid(dot_product(weights, p[0])) + 1e-10)
            - (1 - p[1]) * math.log(1 - sigmoid(dot_product(weights, p[0])) + 1e-10)
        )
    )
    log_loss = temp_rdd.reduce(lambda a, b: a + b) / rdd.count()
    l2_penalty = reg_param * sum(w * w for w in weights[:-1])  # Exclude bias
    return log_loss + l2_penalty

def train_gd(train_rdd, num_features, sc, learning_rate, num_epochs, reg_param, momentum,
             class_weights, early_stop_patience, early_stop_factor):
    """
    Train a logistic regression model using gradient descent with momentum.
    
    Args:
        train_rdd: Training RDD
        num_features: Number of features (excluding bias)
        sc: SparkContext
        learning_rate: Step size for weight updates
        num_epochs: Number of training iterations
        reg_param: L2 regularization parameter
        momentum: Momentum factor for gradient updates
        class_weights: Dictionary of class weights
        early_stop_patience: Epochs to wait before early stopping
        early_stop_factor: Factor to compare loss for early stopping
    
    Returns:
        Tuple of (trained weights, training time)
    """
    weights = [0.0] * (num_features + 1)  # Initialize weights with bias
    velocity = [0.0] * (num_features + 1) # Initialize momentum velocity
    min_loss = float('inf')
    start_time = time.perf_counter()
    count = train_rdd.count()
    
    for epoch in range(num_epochs):
        # Compute gradients across all data in parallel
        gradients = train_rdd.map(
            lambda point: compute_gradient(point, weights, reg_param)
        ).reduce(lambda a, b: [x + y for x, y in zip(a, b)])
        gradients = [g / count for g in gradients]  # Average gradients
        
        # Update velocity using momentum
        velocity = [momentum * v + (1 - momentum) * g for v, g in zip(velocity, gradients)]
        
        # Update weights
        weights = [w - learning_rate * v for w, v in zip(weights, velocity)]
        
        # Compute loss every 5 epochs to reduce computation
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            loss = compute_loss(train_rdd, weights, reg_param, class_weights)
            print(f"Epoch {epoch}, Loss: {loss}")
            
            # Check for early stopping
            if epoch > early_stop_patience and loss > min_loss * early_stop_factor:
                print("Early stopping triggered")
                break
            min_loss = min(min_loss, loss)
    
    training_time = time.perf_counter() - start_time
    return weights, training_time

def evaluate_single_threshold(rdd, weights, threshold):
    """
    Evaluate model performance for a specific threshold.
    
    Args:
        rdd: RDD of (features, label) tuples
        weights: Model weights
        threshold: Classification threshold
    
    Returns:
        Dictionary with accuracy, precision, recall, and F1-score
    """
    predictions = rdd.map(
        lambda p: (p[1], 1 if sigmoid(dot_product(weights, p[0])) > threshold else 0)
    )
    tp = predictions.filter(lambda x: x[0] == 1 and x[1] == 1).count()
    fp = predictions.filter(lambda x: x[0] == 0 and x[1] == 1).count()
    tn = predictions.filter(lambda x: x[0] == 0 and x[1] == 0).count()
    fn = predictions.filter(lambda x: x[0] == 1 and x[1] == 0).count()
    
    # Compute metrics with small constant to avoid division by zero
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def evaluate_model(rdd, weights, thresholds):
    """
    Evaluate model across multiple thresholds.
    
    Args:
        rdd: RDD of (features, label) tuples
        weights: Model weights
        thresholds: List of classification thresholds
    
    Returns:
        Dictionary mapping thresholds to performance metrics
    """
    predictions = rdd.map(
        lambda p: (p[1], sigmoid(dot_product(weights, p[0])))
    )
    results = {}
    
    for threshold in thresholds:
        binary_preds = predictions.map(
            lambda p: (p[0], 1 if p[1] > threshold else 0)
        )
        tp = binary_preds.filter(lambda x: x[0] == 1 and x[1] == 1).count()
        fp = binary_preds.filter(lambda x: x[0] == 0 and x[1] == 1).count()
        tn = binary_preds.filter(lambda x: x[0] == 0 and x[1] == 0).count()
        fn = binary_preds.filter(lambda x: x[0] == 1 and x[1] == 0).count()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        results[threshold] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    
    return results

def grid_search(train_rdd, val_rdd, num_features, sc, param_grid, thresholds, metric='f1_score'):
    """
    Perform grid search over hyperparameters to find the best model.
    
    Args:
        train_rdd: Training RDD
        val_rdd: Validation RDD
        num_features: Number of features
        sc: SparkContext
        param_grid: Dictionary of hyperparameter ranges
        thresholds: List of classification thresholds
        metric: Metric to optimize (default: 'f1_score')
    
    Returns:
        Tuple of (best weights, best score, best params, best training time, best threshold, all results)
    """
    all_results = []
    best_score = 0.0
    best_params = None
    best_weights = None
    best_training_time = 0.0
    best_threshold = None
    
    # Generate all parameter combinations
    param_combinations = list(product(
        param_grid['learning_rates'],
        param_grid['num_epochs'],
        param_grid['reg_params'],
        param_grid['momentums'],
        param_grid['class_weights_list'],
        param_grid['early_stop_patiences'],
        param_grid['early_stop_factors']
    ))
    
    for params in param_combinations:
        (learning_rate, num_epochs, reg_param, momentum,
         class_weights, early_stop_patience, early_stop_factor) = params
        
        print(
            f"\nTrying combination: learning_rate={learning_rate}, num_epochs={num_epochs}, "
            f"reg_param={reg_param}, momentum={momentum}, class_weights={class_weights}, "
            f"early_stop_patience={early_stop_patience}, early_stop_factor={early_stop_factor}"
        )
        
        # Train model with current parameters
        weights, training_time = train_gd(
            train_rdd, num_features, sc, learning_rate, num_epochs,
            reg_param, momentum, class_weights, early_stop_patience, early_stop_factor
        )
        
        # Evaluate on validation set
        val_metrics = evaluate_model(val_rdd, weights, thresholds)
        
        # Select best threshold based on specified metric
        best_threshold_for_model = max(
            val_metrics.keys(), key=lambda t: val_metrics[t][metric]
        )
        score = val_metrics[best_threshold_for_model][metric]
        
        result = {
            'params': params,
            'best_threshold': best_threshold_for_model,
            'val_metrics': val_metrics[best_threshold_for_model],
            'training_time': training_time
        }
        all_results.append(result)
        
        print(
            f"Best Validation {metric} for this model: {score} at threshold {best_threshold_for_model}"
        )
        if score > best_score:
            best_score = score
            best_params = params
            best_weights = weights
            best_training_time = training_time
            best_threshold = best_threshold_for_model
            
    return best_weights, best_score, best_params, best_training_time, best_threshold, all_results

def main(input_path, output_path):
    """
    Main function to load data, train model, and save results.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save output results
    """
    # Initialize Spark session with resource configurations
    spark = SparkSession.builder \
        .appName("Gradient_Descent_RDD_Optimized") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "4") \
        .getOrCreate()
    sc = spark.sparkContext

    # Load and preprocess data
    rdd_data = load_data(input_path, sc)
    normalized_train_rdd, normalized_val_rdd, normalized_test_rdd = preprocessing(rdd_data)
    num_features = len(normalized_train_rdd.first()[0]) - 1  # Exclude bias term

    # Define hyperparameter grid
    param_grid = {
        'learning_rates': [0.1, 0.3, 0.5, 0.7, 1],
        'num_epochs': [30],
        'reg_params': [0.003, 0.001, 0.0007],
        'momentums': [0.9],
        'class_weights_list': [{0: 1.0, 1: 581.0}],
        'early_stop_patiences': [10],
        'early_stop_factors': [1.01]
    }
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Perform grid search to find best model
    (best_weights, best_val_score, best_params, best_training_time,
     best_threshold, all_results) = grid_search(
        normalized_train_rdd, normalized_val_rdd,
        num_features, sc, param_grid, thresholds
    )
    
    # Evaluate best model on test set
    test_metrics = evaluate_single_threshold(normalized_test_rdd, best_weights, best_threshold)

    # Format results for output
    formatted_results = []
    for i, result in enumerate(all_results):
        params_str = (
            f"learning_rate={result['params'][0]}, num_epochs={result['params'][1]}, "
            f"reg_param={result['params'][2]}, momentum={result['params'][3]}, "
            f"class_weights={result['params'][4]}, early_stop_patience={result['params'][5]}, "
            f"early_stop_factor={result['params'][6]}, Best_Threshold={result['best_threshold']}"
        )
        metrics_str = (
            f"Training_Time: {result['training_time']:.2f}s, "
            f"Val_Acc: {result['val_metrics']['accuracy']:.4f}, "
            f"Val_Prec: {result['val_metrics']['precision']:.4f}, "
            f"Val_Rec: {result['val_metrics']['recall']:.4f}, "
            f"Val_F1: {result['val_metrics']['f1_score']:.4f}"
        )
        formatted_results.append(f"Run {i+1}: Params: {params_str}, Metrics: {metrics_str}")

    # Summarize best model results
    best_results = [
        f"\nBest Model Results:",
        f"Best Weights (including bias): {best_weights}",
        f"Best Validation F1-score: {best_val_score}",
        f"Best Training Time: {best_training_time:.2f} seconds",
        f"Best Threshold: {best_threshold}",
        f"Test_Acc: {test_metrics['accuracy']:.4f}, "
        f"Test_Prec: {test_metrics['precision']:.4f}, "
        f"Test_Rec: {test_metrics['recall']:.4f}, "
        f"Test_F1: {test_metrics['f1_score']:.4f}",
        f"Best Parameters: learning_rate={best_params[0]}, num_epochs={best_params[1]}, "
        f"reg_param={best_params[2]}, momentum={best_params[3]}, "
        f"class_weights={best_params[4]}, early_stop_patience={best_params[5]}, "
        f"early_stop_factor={best_params[6]}, threshold={best_threshold}"
    ]

    # Save results to output path
    final_output = formatted_results + best_results
    sc.parallelize(final_output).coalesce(1).saveAsTextFile(output_path)
    print("Results saved to:", output_path)

    # Clean up Spark session
    spark.stop()

if __name__ == "__main__":
    """
    Entry point for the script.
    
    Expects command-line arguments: input_path and output_path.
    """
    if len(sys.argv) != 3:
        print("Usage: spark-submit script.py <input_path> <output_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
