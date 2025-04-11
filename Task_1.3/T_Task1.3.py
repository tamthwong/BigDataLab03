import sys
from pyspark.sql import SparkSession
import math
from itertools import product
import random
import time

def z_score_normalize(rdd, mean_values, std_values):
    return rdd.map(lambda row: (tuple([(x - mean) / std if std != 0 else 0 for x, mean, std in zip(row[0], mean_values, std_values)] + [1.0]), row[1]))

def compute_mean_std(train_rdd):
    count = train_rdd.count()
    sum_vector = train_rdd.map(lambda x: x[0]).reduce(lambda a, b: [x + y for x, y in zip(a, b)])
    mean_vector = [x / count for x in sum_vector]
    sum_squared_diff = train_rdd.map(lambda x: [(xi - mi) ** 2 for xi, mi in zip(x[0], mean_vector)]) \
                               .reduce(lambda a, b: [x + y for x, y in zip(a, b)])
    std_vector = [math.sqrt(x / count + 1e-10) for x in sum_squared_diff]
    return mean_vector, std_vector

def parse_line(line):
    parts = line.split(",")
    features = tuple(list(map(float, parts[1:-1])))
    raw_label = parts[-1].replace('"', '')
    label = int(raw_label)
    return (features, label)

def load_data(input_path, sc):
    lines = sc.textFile(input_path)
    header = lines.first()
    data = lines.filter(lambda line: line != header)
    rdd_data = data.map(parse_line)
    return rdd_data

def preprocessing(rdd_data):
    rdd_data = rdd_data.distinct()
    rdd_data = rdd_data.filter(lambda x: all(not math.isnan(f) for f in x[0]) and not math.isnan(x[1]))
    train_rdd, val_rdd, test_rdd = rdd_data.randomSplit([0.7, 0.1, 0.2], seed=42)
    mean_values, std_values = compute_mean_std(train_rdd)
    normalized_train_rdd = z_score_normalize(train_rdd, mean_values, std_values)
    normalized_val_rdd = z_score_normalize(val_rdd, mean_values, std_values)
    normalized_test_rdd = z_score_normalize(test_rdd, mean_values, std_values)
    return normalized_train_rdd, normalized_val_rdd, normalized_test_rdd

def dot_product(weights, features):
    return sum(w * f for w, f in zip(weights, features))

def sigmoid(z):
    if z > 100:
        return 1.0
    elif z < -100:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))

def compute_gradient(point, weights, reg_param):
    features, label = point
    pred = sigmoid(dot_product(weights, features))
    grad = [f * (pred - label) for f in features]
    return [g + reg_param * w if i < len(weights) - 1 else g for i, (g, w) in enumerate(zip(grad, weights))]

def compute_loss(rdd, weights, reg_param, class_weights):
    temp_rdd = rdd.map(lambda p: class_weights[p[1]] * (
        -p[1] * math.log(sigmoid(dot_product(weights, p[0])) + 1e-10) 
        - (1 - p[1]) * math.log(1 - sigmoid(dot_product(weights, p[0])) + 1e-10)))
    log_loss = temp_rdd.reduce(lambda a, b: a + b) / temp_rdd.count()
    l2_penalty = reg_param * sum(w * w for w in weights[:-1])
    return log_loss + l2_penalty

def train_sgd(train_rdd, num_features, sc, learning_rate, num_epochs, batch_size, 
              reg_param, momentum, class_weights, early_stop_patience, early_stop_factor):
    weights = [0.0] * (num_features + 1)
    velocity = [0.0] * (num_features + 1)
    min_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        batch_rdd = train_rdd.takeSample(withReplacement=False, num=batch_size, seed=epoch)
        gradients = [0.0] * (num_features + 1)
        for point in batch_rdd:
            grad = compute_gradient(point, weights, reg_param)
            gradients = [g + dg for g, dg in zip(gradients, grad)]
        
        velocity = [momentum * v + (1 - momentum) * (g / batch_size) for v, g in zip(velocity, gradients)]
        weights = [w - learning_rate * v for w, v in zip(weights, velocity)]
        
        loss = compute_loss(train_rdd, weights, reg_param, class_weights)
        print(f"Epoch {epoch}, Loss: {loss}")
        
        if epoch > early_stop_patience and loss > min_loss * early_stop_factor:
            print("Early stopping triggered")
            break
        min_loss = min(min_loss, loss)
    
    training_time = time.time() - start_time
    return weights, training_time

def evaluate_model(rdd, weights, thresholds):
    predictions = rdd.map(lambda p: (p[1], sigmoid(dot_product(weights, p[0]))))
    results = {}
    
    for threshold in thresholds:
        binary_preds = predictions.map(lambda p: (p[0], 1 if p[1] > threshold else 0))
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

def evaluate_single_threshold(rdd, weights, threshold):
    predictions = rdd.map(lambda p: (p[1], 1 if sigmoid(dot_product(weights, p[0])) > threshold else 0))
    tp = predictions.filter(lambda x: x[0] == 1 and x[1] == 1).count()
    fp = predictions.filter(lambda x: x[0] == 0 and x[1] == 1).count()
    tn = predictions.filter(lambda x: x[0] == 0 and x[1] == 0).count()
    fn = predictions.filter(lambda x: x[0] == 1 and x[1] == 0).count()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}

def grid_search(train_rdd, val_rdd, num_features, sc, param_grid, thresholds, metric='f1_score'):
    all_results = []
    best_score = 0.0
    best_params = None
    best_weights = None
    best_training_time = 0.0
    best_threshold = None
    
    param_combinations = list(product(
        param_grid['learning_rates'],
        param_grid['num_epochs'],
        param_grid['batch_sizes'],
        param_grid['reg_params'],
        param_grid['momentums'],
        param_grid['class_weights_list'],
        param_grid['early_stop_patiences'],
        param_grid['early_stop_factors']
    ))
    
    for params in param_combinations:
        (learning_rate, num_epochs, batch_size, reg_param, momentum, 
         class_weights, early_stop_patience, early_stop_factor) = params
        
        print(f"\nTrying combination: learning_rate={learning_rate}, num_epochs={num_epochs}, "
              f"batch_size={batch_size}, reg_param={reg_param}, momentum={momentum}, "
              f"class_weights={class_weights}, early_stop_patience={early_stop_patience}, "
              f"early_stop_factor={early_stop_factor}")
        
        weights, training_time = train_sgd(
            train_rdd, num_features, sc, learning_rate, num_epochs, batch_size,
            reg_param, momentum, class_weights, early_stop_patience, early_stop_factor
        )
        val_metrics = evaluate_model(val_rdd, weights, thresholds)
        
        # Find best threshold for this model based on the specified metric
        best_threshold_for_model = max(val_metrics.keys(), key=lambda t: val_metrics[t][metric])
        score = val_metrics[best_threshold_for_model][metric]
        
        result = {
            'params': params,
            'best_threshold': best_threshold_for_model,
            'val_metrics': val_metrics[best_threshold_for_model],
            'training_time': training_time
        }
        all_results.append(result)
        
        print(f"Best Validation {metric} for this model: {score} at threshold {best_threshold_for_model}")
        if score > best_score:
            best_score = score
            best_params = params
            best_weights = weights
            best_training_time = training_time
            best_threshold = best_threshold_for_model
            
    return best_weights, best_score, best_params, best_training_time, best_threshold, all_results

def main(input_path, output_path):
    spark = SparkSession.builder \
        .appName("SGD_Logistic_Regression_RDD_with_Momentum") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "4") \
        .getOrCreate()
    sc = spark.sparkContext

    rdd_data = load_data(input_path, sc)
    normalized_train_rdd, normalized_val_rdd, normalized_test_rdd = preprocessing(rdd_data)
    num_features = len(normalized_train_rdd.first()[0]) - 1

    param_grid = {
        'learning_rates': [0.1, 0.3, 0.5, 0.7, 1],
        'num_epochs': [30],
        'batch_sizes': [600],
        'reg_params': [0.003, 0.001, 0.0007],
        'momentums': [0.9],
        'class_weights_list': [{0: 1.0, 1: 581.0}],
        'early_stop_patiences': [10],
        'early_stop_factors': [1.001]
    }
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    (best_weights, best_val_score, best_params, best_training_time, 
     best_threshold, all_results) = grid_search(normalized_train_rdd, normalized_val_rdd, 
                                              num_features, sc, param_grid, thresholds)
    
    test_metrics = evaluate_single_threshold(normalized_test_rdd, best_weights, best_threshold)

    # Format all results
    formatted_results = []
    for i, result in enumerate(all_results):
        params_str = (f"learning_rate={result['params'][0]}, num_epochs={result['params'][1]}, "
                     f"batch_size={result['params'][2]}, reg_param={result['params'][3]}, "
                     f"momentum={result['params'][4]}, class_weights={result['params'][5]}, "
                     f"early_stop_patience={result['params'][6]}, early_stop_factor={result['params'][7]}, "
                     f"Best_Threshold={result['best_threshold']}")
        metrics_str = (f"Training_Time: {result['training_time']:.2f}s, "
                      f"Val_Acc: {result['val_metrics']['accuracy']:.4f}, "
                      f"Val_Prec: {result['val_metrics']['precision']:.4f}, "
                      f"Val_Rec: {result['val_metrics']['recall']:.4f}, "
                      f"Val_F1: {result['val_metrics']['f1_score']:.4f}")
        formatted_results.append(f"Run {i+1}: Params: {params_str}, Metrics: {metrics_str}")

    # Add best model results
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
        f"batch_size={best_params[2]}, reg_param={best_params[3]}, momentum={best_params[4]}, "
        f"class_weights={best_params[5]}, early_stop_patience={best_params[6]}, "
        f"early_stop_factor={best_params[7]}, threshold={best_threshold}"
    ]

    final_output = formatted_results + best_results
    sc.parallelize(final_output).coalesce(1).saveAsTextFile(output_path)
    print("Results saved to:", output_path)

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit script.py <input_path> <output_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])