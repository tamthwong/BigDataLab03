from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, explode
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType
import os

# --- MONGODB CONFIGURATION ---

# MongoDB URI from env variable (set this before running)
# Example: mongodb+srv://user:password@cluster.mongodb.net/mydb?retryWrites=true&w=majority
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://bigdataStore:2ksGGNqOZM2omRek@bigdata.r0uwyz9.mongodb.net/?retryWrites=true&w=majority&appName=bigdata")
MONGO_DB = os.getenv("MONGO_DB", "btc_analysis")

# --- INITIALIZE SPARK SESSION ---
spark = SparkSession.builder \
    .appName("BTCPriceZScoreStreaming") \
    .config("spark.mongodb.output.uri", MONGO_URI) \
    .config("spark.jars.packages",
            "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0,"
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# --- DEFINE SCHEMA FOR KAFKA MESSAGE ---
zscore_schema = StructType([
    StructField("window", StringType(), False),
    StructField("zscore_price", FloatType(), False)
])

schema = StructType([
    StructField("timestamp", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("zscores", ArrayType(zscore_schema), False)
])

# --- READ FROM KAFKA ---
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "btc-price-zscore") \
    .option("startingOffsets", "latest") \
    .load()

# --- PARSE JSON AND EXPLODE zscores ARRAY ---
parsed_df = kafka_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select(
    col("data.timestamp").cast("timestamp").alias("timestamp"),
    col("data.symbol").alias("symbol"),
    col("data.zscores").alias("zscores")
)

exploded_df = parsed_df.select(
    col("timestamp"),
    col("symbol"),
    explode(col("zscores")).alias("zscore")
).select(
    col("timestamp"),
    col("symbol"),
    col("zscore.window").alias("window"),
    col("zscore.zscore_price").alias("zscore_price")
)

# --- DEFINE TIME WINDOWS TO SEPARATE COLLECTIONS ---
windows = ["30s", "1m", "5m", "15m", "30m", "1h"]

# --- CREATE A WRITE STREAM FOR EACH WINDOW TO DIFFERENT COLLECTIONS ---
queries = []

for window in windows:
    window_df = exploded_df.filter(col("window") == window)

    try:
        query = window_df.writeStream \
            .format("mongo") \
            .option("spark.mongodb.output.uri", MONGO_URI) \
            .option("database", MONGO_DB) \
            .option("collection", f"btc_price_zscore_{window}") \
            .option("checkpointLocation", f"/tmp/spark_checkpoint_btc_zscore_{window}") \
            .outputMode("append") \
            .start()

        print(f"[INFO] Started stream for window: {window}")
        queries.append(query)

    except Exception as e:
        print(f"[ERROR] Failed to start stream for window {window}: {str(e)}")

# --- LOG STATUS OF ALL STREAMS PERIODICALLY (OPTIONAL) ---
for query in queries:
    print(f"[STATUS] Stream {query.id} active: {query.isActive}")

# --- WAIT FOR TERMINATION OF ANY STREAM ---
spark.streams.awaitAnyTermination()
