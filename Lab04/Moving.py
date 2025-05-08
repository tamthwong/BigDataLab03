from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, avg, stddev, window, from_json, to_timestamp, collect_list, struct, lit, when, size, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import os

class SparkConfig:
    """Handles Spark session creation and configuration."""
    def __init__(self, app_name: str, kafka_bootstrap_servers: str):
        self.app_name = app_name
        self.kafka_bootstrap_servers = kafka_bootstrap_servers

    def create_spark_session(self) -> SparkSession:
        """Creates and configures a Spark session with Kafka integration."""
        return SparkSession.builder \
            .appName(self.app_name) \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
            .getOrCreate()

class KafkaReader:
    """Manages reading from Kafka topics."""
    def __init__(self, spark: SparkSession, bootstrap_servers: str):
        self.spark = spark
        self.bootstrap_servers = bootstrap_servers
        self.raw_schema = StructType([
            StructField("symbol", StringType(), True),
            StructField("price", DoubleType(), True),
            StructField("timestamp", StringType(), True)
        ])
        self.inter_schema = StructType([
            StructField("timestamp", TimestampType(), True),
            StructField("symbol", StringType(), True),
            StructField("window", StringType(), True),
            StructField("avg_price", DoubleType(), True),
            StructField("std_price", DoubleType(), True)
        ])

    def read_raw_stream(self, topic: str) -> DataFrame:
        """Reads and parses raw streaming data from a Kafka topic."""
        raw_df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "earliest") \
            .load()

        parsed_df = raw_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), self.raw_schema).alias("data")) \
            .select("data.*") \
            .withColumn("timestamp", to_timestamp(col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX")) \
            .withWatermark("timestamp", "10 seconds")
        return parsed_df

    def read_intermediate_stream(self, topic: str) -> DataFrame:
        """Reads and parses intermediate window stats from a Kafka topic."""
        interm_df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "earliest") \
            .load() \
            .selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), self.inter_schema).alias("data")) \
            .select("data.*") \
            .withWatermark("timestamp", "10 seconds")
        return interm_df

class KafkaWriter:
    """Manages writing to Kafka topics."""
    def __init__(self, bootstrap_servers: str, checkpoint_dir: str):
        self.bootstrap_servers = bootstrap_servers
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir.replace("file://", ""), exist_ok=True)

    def write_stream(self, df: DataFrame, topic: str, checkpoint_subdir: str) -> None:
        """Writes a DataFrame to a Kafka topic."""
        kafka_df = df.selectExpr(
            "CAST(symbol AS STRING) AS key",
            "to_json(struct(*)) AS value"
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_subdir)
        os.makedirs(checkpoint_path.replace("file://", ""), exist_ok=True)
        kafka_df.writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("topic", topic) \
            .option("checkpointLocation", checkpoint_path) \
            .outputMode("append") \
            .start()

class WindowStatsProcessor:
    """Computes windowed statistics (average and standard deviation)."""
    def __init__(self):
        self.window_map = {
            "30 seconds": "30s",
            "1 minute": "1m",
            "5 minutes": "5m",
            "15 minutes": "15m",
            "30 minutes": "30m",
            "1 hour": "1h"
        }
        self.windows = ["30 seconds", "1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour"]

    def compute_stats(self, df: DataFrame, window_duration: str) -> DataFrame:
        """Computes average and standard deviation for a given window duration."""
        abbr = self.window_map.get(window_duration, window_duration)
        stats_df = df.groupBy(
            window(col("timestamp"), window_duration).alias("win"),
            col("symbol")
        ) \
            .agg(
                avg("price").alias("avg_price"),
                stddev("price").alias("std_price"),
                count("price").alias("count")
            ) \
            .select(
                col("win.end").alias("timestamp"),
                col("symbol"),
                lit(abbr).alias("window"),
                when(col("avg_price").isNotNull(), col("avg_price")).otherwise(lit(None)).alias("avg_price"),
                when(col("count") == 1, lit(0.0)).otherwise(
                    when(col("std_price").isNotNull(), col("std_price")).otherwise(lit(None))
                ).alias("std_price")
            )
        return stats_df

class StatsAggregator:
    """Aggregates windowed statistics into the final output format."""
    def aggregate(self, df: DataFrame) -> DataFrame:
        """Groups window stats by timestamp and symbol into an array."""
        return df.groupBy("timestamp", "symbol") \
            .agg(collect_list(struct("window", "avg_price", "std_price")).alias("windows"))

class PipelineOrchestrator:
    """Coordinates the streaming pipeline."""
    def __init__(self,
                 kafka_bootstrap_servers: str,
                 input_topic: str,
                 output_topic: str,
                 checkpoint_dir: str):
        self.spark_config = SparkConfig("Transformer", kafka_bootstrap_servers)
        self.spark = self.spark_config.create_spark_session()
        self.kafka_reader = KafkaReader(self.spark, kafka_bootstrap_servers)
        self.kafka_writer = KafkaWriter(kafka_bootstrap_servers, checkpoint_dir)
        self.window_processor = WindowStatsProcessor()
        self.aggregator = StatsAggregator()
        self.input_topic = input_topic
        self.intermediate_topic = output_topic + "-wins"
        self.output_topic = output_topic

    def run(self):
        """Runs the streaming pipeline."""
        print("Starting BTC Price Moving Average app...")

        # Read raw data
        df = self.kafka_reader.read_raw_stream(self.input_topic)
        print("Input schema:")
        df.printSchema()

        # Compute windowed statistics
        stats = None
        for w in self.window_processor.windows:
            win_df = self.window_processor.compute_stats(df, w)
            stats = win_df if stats is None else stats.unionByName(win_df)

        # Write intermediate stats
        self.kafka_writer.write_stream(stats, self.intermediate_topic, "moving_wins")

        # Read and aggregate intermediate stats
        interm_df = self.kafka_reader.read_intermediate_stream(self.intermediate_topic)
        grouped_df = self.aggregator.aggregate(interm_df)

        # Write final output
        self.kafka_writer.write_stream(grouped_df, self.output_topic, "moving_grouped")

        # Await termination
        self.spark.streams.awaitAnyTermination()

def main():
    checkpoint_path = "/tmp/spark_checkpoint/btc-price-moving"
    pipeline = PipelineOrchestrator(
        kafka_bootstrap_servers="localhost:9092",
        input_topic="btc-price",
        output_topic="btc-price-moving",
        checkpoint_dir=checkpoint_path
    )
    pipeline.run()

if __name__ == "__main__":
    main()