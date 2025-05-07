from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, expr, collect_list, struct, to_json
from pyspark.sql.types import StructType, StringType, TimestampType, StructField, DoubleType

# Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("BTCWindowOOP") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# Định nghĩa schema cho dữ liệu Kafka
schema = StructType([
    StructField("symbol", StringType(), False),
    StructField("price", DoubleType(), False),
    StructField("timestamp", StringType(), False)
])

# Đọc dữ liệu từ Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "btc-price") \
    .option("startingOffsets", "latest") \
    .load()

# Parse dữ liệu JSON từ Kafka
parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select(
        col("data.symbol").alias("symbol"),
        col("data.price").alias("price"),
        col("data.timestamp").cast("timestamp").alias("timestamp")
    ).withWatermark("timestamp", "25 seconds")

# Self-join trong khoảng thời gian 20 giây
joined_df = parsed_df.alias("base").join(
    parsed_df.alias("future"),
    (col("base.symbol") == col("future.symbol")) &
    (col("future.timestamp") > col("base.timestamp")) &
    (col("future.timestamp") <= expr("base.timestamp + INTERVAL 20 SECONDS")),
    "inner"
).select(
    col("base.symbol").alias("symbol"),
    col("base.timestamp").alias("base_timestamp"),
    col("base.price").alias("base_price"),
    col("future.timestamp").alias("future_timestamp"),
    col("future.price").alias("future_price")
)

# Gom nhóm dữ liệu thành mảng `next`
aggregated = joined_df.groupBy("symbol", "base_timestamp", "base_price") \
    .agg(
        collect_list(
            struct(
                col("future_timestamp").alias("timestamp"),
                col("future_price").alias("price")
            )
        ).alias("next")
    )



from pyspark.sql.functions import udf, struct, to_json, lit
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F

# UDF cho thời gian giảm
def min_decrease_time(base_price, base_timestamp, next_list):
    try:
        for entry in sorted(next_list, key=lambda x: x['timestamp']):
            if entry['price'] < base_price:
                return float((entry['timestamp'] - base_timestamp).total_seconds())
        return 20.0
    except:
        return 20.0

# UDF cho thời gian tăng
def min_increase_time(base_price, base_timestamp, next_list):
    try:
        for entry in sorted(next_list, key=lambda x: x['timestamp']):
            if entry['price'] > base_price:
                return float((entry['timestamp'] - base_timestamp).total_seconds())
        return 20.0
    except:
        return 20.0

# Đăng ký UDF
decrease_udf = udf(min_decrease_time, DoubleType())
increase_udf = udf(min_increase_time, DoubleType())

# Tính thời gian tăng/giảm
final_df = aggregated.withColumn("time_to_decrease", decrease_udf("base_price", "base_timestamp", "next")) \
                     .withColumn("time_to_increase", increase_udf("base_price", "base_timestamp", "next"))

# Tạo 2 dòng riêng biệt cho higher và lower
higher_df = final_df.select(
    to_json(
        struct(
            col("base_timestamp").cast("string").alias("timestamp"),
            col("time_to_increase").alias("higher_window")
        )
    ).alias("value")
)

lower_df = final_df.select(
    to_json(
        struct(
            col("base_timestamp").cast("string").alias("timestamp"),
            col("time_to_decrease").alias("lower_window")
        )
    ).alias("value")
)

from pyspark.sql.functions import to_json, struct, col

# Tạo JSON cho higher_window
higher_json_df = final_df.select(
    to_json(
        struct(
            col("base_timestamp").cast("string").alias("timestamp"),
            col("time_to_increase").alias("higher_window")
        )
    ).alias("value")
)

# Tạo JSON cho lower_window
lower_json_df = final_df.select(
    to_json(
        struct(
            col("base_timestamp").cast("string").alias("timestamp"),
            col("time_to_decrease").alias("lower_window")
        )
    ).alias("value")
)

# Ghi higher_window vào Kafka topic "btc-price-higher"
higher_query = higher_json_df.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "btc-price-higher") \
    .option("checkpointLocation", "/tmp/checkpoints/btc-price-higher") \
    .outputMode("append") \
    .start()

# Ghi lower_window vào Kafka topic "btc-price-lower"
lower_query = lower_json_df.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "btc-price-lower") \
    .option("checkpointLocation", "/tmp/checkpoints/btc-price-lower") \
    .outputMode("append") \
    .start()

# Chờ cả hai luồng kết thúc
higher_query.awaitTermination()
lower_query.awaitTermination()



