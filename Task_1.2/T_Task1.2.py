from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.mllib.linalg import Vectors
import sys
import time, math

def evaluate_model(model, rdd):
    predictions_and_labels = rdd.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    
    # Calculate Accuracy
    correct = predictions_and_labels.filter(lambda pl: pl[0] == pl[1]).count()
    total = rdd.count()
    accuracy = correct / total
    
    # Calculate Cross-Entropy Loss
    def compute_ce(pred, label):
        # Clip pred to avoid log(0)
        eps = 1e-15
        pred = max(min(pred, 1 - eps), eps)
        return -label * math.log(pred) - (1 - label) * math.log(1 - pred)
    
    # Calculate MBE (Mean Bias Error)
    # bias_sum = predictions_and_labels.map(lambda pl: pl[0] - pl[1]).reduce(lambda a, b: a + b)
    # mbe = bias_sum / total
    ce_sum = predictions_and_labels.map(lambda pl: compute_ce(pl[0], pl[1])).reduce(lambda a, b: a + b)
    ce = ce_sum / total
    
    # Calculate TP, FP, FN
    tp = predictions_and_labels.filter(lambda pl: pl[0] == 1.0 and pl[1] == 1.0).count()
    fp = predictions_and_labels.filter(lambda pl: pl[0] == 1.0 and pl[1] == 0.0).count()
    fn = predictions_and_labels.filter(lambda pl: pl[0] == 0.0 and pl[1] == 1.0).count()

    # Calculate Precision, Recall, F1 Score
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    return ce, accuracy, precision, recall, f1_score

def convert_ml_vector_to_mllib(v):
    if isinstance(v, DenseVector):
        return Vectors.dense(v.toArray())
    elif isinstance(v, SparseVector):
        return Vectors.sparse(v.size, v.indices, v.values)
    else:
        raise ValueError("Unsupported kind of vector: {}".format(type(v)))

def RDDLogRegression(spark, input_file, output_file):
    results = []
    
    df = spark.read.csv(input_file, header=True, inferSchema=True)
    input_cols = [col for col in df.columns if col != 'Class' and col != 'Time']
    df = df.na.drop()
    
    assembler = VectorAssembler(inputCols=input_cols, outputCol="raw_features")
    assembled_data = assembler.transform(df).select("raw_features", "Class")

    train, validation, test = assembled_data.randomSplit([0.7, 0.1, 0.2], seed=42)
    
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    
    scaler_model = scaler.fit(train)
    
    scaled_train = scaler_model.transform(train).select("features", "Class")
    scaled_valid = scaler_model.transform(validation).select("features", "Class")
    scaled_test = scaler_model.transform(test).select("features", "Class")

    rdd_train = scaled_train.rdd.map(lambda row: LabeledPoint(row.Class, convert_ml_vector_to_mllib(row.features)))
    rdd_valid = scaled_valid.rdd.map(lambda row: LabeledPoint(row.Class, convert_ml_vector_to_mllib(row.features)))
    rdd_test = scaled_test.rdd.map(lambda row: LabeledPoint(row.Class, convert_ml_vector_to_mllib(row.features)))

    model_candidate = {
        'iterations': [50, 75, 100, 150, 200],
        'step': [15, 20, 25, 30, 35]
    }
    
    best_model = None
    best_f1_score = 0.0
    best_step = 0.0
    best_iterations = 0

    for step in model_candidate['step']:
        for iterations in model_candidate['iterations']:
            print(f"-------------------->>Iterations: {iterations}, Step size: {step}")
            
            # Training time
            start_time = time.time()
            model = LogisticRegressionWithSGD.train(rdd_train, iterations=iterations, step=step)
            end_time = time.time()
            print(f"----------{iterations}, {step}----------\nTraining time: {(end_time - start_time):.2f} seconds")
            
            # Training evaluation metrics
            train_ce, train_accuracy, train_precision, train_recall, train_f1_score = evaluate_model(model, rdd_train)
            print(f"----------\nTrain CE: {train_ce:.4f}\nTrain accuracy: {train_accuracy:.4f}\nTrain precision: {train_precision:.4f}\nTrain recall: {train_recall:.4f}\nTrain F1-score: {train_f1_score:.4f}")
            
            # Validation evaluation metrics
            valid_ce, valid_accuracy, valid_precision, valid_recall, valid_f1_score = evaluate_model(model, rdd_valid)
            print(f"----------\nValidation CE: {valid_ce:.4f}\nValidation accuracy: {valid_accuracy:.4f}\nValidation precision: {valid_precision:.4f}\nValidation recall: {valid_recall:.4f}\nValidation F1-score: {valid_f1_score:.4f}")
            
            if valid_f1_score > best_f1_score:
                best_f1_score = valid_f1_score
                best_model = model 
                best_step = step
                best_iterations = iterations
                
            results.append(f"iterations={iterations}, stepSize={step}, TrainCE={train_ce:.4f}, TrainAcc={train_accuracy:.4f}, TrainPrec={train_precision:.4f}, TrainRec={train_recall:.4f}, TrainF1={train_f1_score:.4f}, ValCE={valid_ce:.4f}, ValAcc={valid_accuracy:.4f}, ValPrec={valid_precision:.4f}, ValRec={valid_recall:.4f}, ValF1={valid_f1_score:.4f}, Time={(end_time - start_time):.2f}s")

    test_ce, test_accuracy, test_precision, test_recall, test_f1_score = evaluate_model(model, rdd_test)

    print(f"Best parameters: Iterations: {best_iterations}, Step size: {best_step}, CE: {test_ce:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1_score:.4f}")
    results.append(f"Best parameters: Iterations: {best_iterations}, Step size: {best_step}, CE: {test_ce:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1_score:.4f}")
    results.append(f"Coefficients: {best_model.weights.toArray().tolist()}")

    spark.sparkContext.parallelize(results).coalesce(1).saveAsTextFile(output_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: LogisticRegression <input_file> <output_file>")
        sys.exit(-1)
        
    spark = SparkSession.builder.appName("LogisticRegression").getOrCreate()

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    RDDLogRegression(spark, input_file, output_file)

    spark.stop()