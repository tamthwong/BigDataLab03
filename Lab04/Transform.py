from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("MovingStats") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .getOrCreate()

# Định nghĩa schema
schema = StructType([
    StructField("symbol", StringType(), False),
    StructField("price", DoubleType(), False),
    StructField("timestamp", StringType(), False)
])

# Đọc từ Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "btc-price") \
    .option("startingOffsets", "latest") \
    .load() \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*") \
    .withColumn("event_time", to_timestamp(col("timestamp")))

# Danh sách các cửa sổ thời gian
windows = [
    ("30 seconds", "window_30s", "30s"),
    ("1 minute", "window_1m", "1m")
]

# Tính toán cho từng cửa sổ và gộp kết quả
window_dfs = []
for window_duration, window_alias, window_label in windows:
    window_df = df \
        .withWatermark("event_time", "10 seconds") \
        .groupBy(
            col("symbol"),
            window(col("event_time"), window_duration).alias(window_alias)
        ) \
        .agg(
            avg("price").alias(f"avg_price_{window_label}"),
            stddev("price").alias(f"std_price_{window_label}")
        ) \
        .select(
            col("symbol"),
            to_timestamp(col(f"{window_alias}.end")).alias("timestamp"),
            lit(window_label).alias("window"),
            col(f"avg_price_{window_label}").alias("avg_price"),
            col(f"std_price_{window_label}").alias("std_price")
        )
    window_dfs.append(window_df)

# Gộp tất cả các DataFrame theo symbol và timestamp
combined_df = window_dfs[0]
for window_df in window_dfs[1:]:
    combined_df = combined_df.unionByName(window_df)

# Áp dụng watermark trước aggregation cuối cùng
combined_df = combined_df.withWatermark("timestamp", "10 seconds")

# Nhóm lại để tạo cấu trúc mảng windows
output = combined_df \
    .groupBy("symbol", "timestamp") \
    .agg(
        collect_list(struct("window", "avg_price", "std_price")).alias("windows")
    )

# Ghi vào Kafka topic btc-price-moving
query = output \
    .select(to_json(struct("*")).alias("value")) \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "btc-price-moving") \
    .option("checkpointLocation", "/tmp/spark-checkpoint-moving-1") \
    .outputMode("append") \
    .start()

query.awaitTermination()