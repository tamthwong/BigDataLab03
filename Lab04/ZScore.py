from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json, to_timestamp, explode, struct, lit, when, collect_list, expr, udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, ArrayType
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
    def __init__(self, spark: SparkSession, bootstrap_servers: str, watermark_duration: str):
        self.spark = spark
        self.bootstrap_servers = bootstrap_servers
        self.watermark_duration = watermark_duration  # Added watermark duration
        self.price_schema = StructType([
            StructField("symbol", StringType(), True),
            StructField("price", DoubleType(), True),
            StructField("timestamp", StringType(), True)
        ])
        self.moving_schema = StructType([
            StructField("timestamp", TimestampType(), True),
            StructField("symbol", StringType(), True),
            StructField("windows", ArrayType(StructType([
                StructField("window", StringType(), True),
                StructField("avg_price", DoubleType(), True),
                StructField("std_price", DoubleType(), True)
            ])), True)
        ])
        self.stats_schema = StructType([
            StructField("timestamp", TimestampType(), True),
            StructField("symbol", StringType(), True),
            StructField("window", StringType(), True),
            StructField("avg_price", DoubleType(), True),
            StructField("std_price", DoubleType(), True)
        ])

    def read_price_stream(self, topic: str) -> DataFrame:
        """Reads and parses raw price data from btc-price topic."""
        raw_df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "earliest") \
            .load()

        parsed_df = raw_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), self.price_schema).alias("data")) \
            .select(
                "data.symbol",
                "data.price",
                to_timestamp(col("data.timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX").alias("timestamp")
            ) \
            .withWatermark("timestamp", self.watermark_duration) # Use the parameter
        return parsed_df

    def read_moving_stream(self, topic: str) -> DataFrame:
        """Reads and parses moving statistics from btc-price-moving topic."""
        raw_df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "earliest") \
            .load()

        parsed_df = raw_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), self.moving_schema).alias("data")) \
            .select(
                "data.timestamp",
                "data.symbol",
                "data.windows"
            ) \
            .withWatermark("timestamp", self.watermark_duration)  # Use the parameter
        return parsed_df

    def read_intermediate_stream(self, topic: str) -> DataFrame:
        """Reads and parses intermediate stats from btc-price-zscore-wins topic."""
        raw_df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "earliest") \
            .load()

        parsed_df = raw_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), self.stats_schema).alias("data")) \
            .select(
                "data.timestamp",
                "data.symbol",
                "data.window",
                "data.avg_price",
                "data.std_price"
            ) \
            .withWatermark("timestamp", self.watermark_duration)  # Use the parameter
        return parsed_df

class KafkaWriter:
    """Manages writing to Kafka topics."""
    def __init__(self, bootstrap_servers: str, checkpoint_dir: str):
        self.bootstrap_servers = bootstrap_servers
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir.replace("file://", ""), exist_ok=True)

    def write_stream(self, df: DataFrame, topic: str, checkpoint_subdir: str, output_mode: str = "append") -> None:
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
            .outputMode(output_mode) \
            .start()

class ZScoreProcessor:
    """Computes Z-scores for prices based on moving statistics."""
    def process_moving_to_stats(self, moving_df: DataFrame) -> DataFrame:
        """Explodes windows from btc-price-moving to btc-price-zscore-wins format."""
        stats_df = moving_df.select(
            col("timestamp"),
            col("symbol"),
            explode(col("windows")).alias("window_struct")
        ).select(
            col("timestamp"),
            col("symbol"),
            col("window_struct.window").alias("window"),
            col("window_struct.avg_price").alias("avg_price"),
            col("window_struct.std_price").alias("std_price")
        )
        return stats_df

    def compute_zscores(self, price_df: DataFrame, stats_df: DataFrame) -> DataFrame:
        """Joins price and stats by window range, computes Z-scores, groups by price timestamp."""
        # Parse window duration from stats_df.window (e.g., '30s' -> 30, '1m' -> 60)
        stats_df = stats_df.select(
            col("timestamp"),
            col("symbol"),
            col("window"),
            col("avg_price"),
            col("std_price"),
            expr("""
                CASE
                    WHEN window = '30s' THEN 30
                    WHEN window = '1m' THEN 60
                    WHEN window = '5m' THEN 300
                    WHEN window = '15m' THEN 900
                    WHEN window = '30m' THEN 1800
                    WHEN window = '1h' THEN 3600
                    ELSE 0
                END
            """).cast("long").alias("window_seconds")
        ).alias("stats")

        # Filter out nulls to prevent null-related errors
        stats_df = stats_df.filter(
            col("stats.timestamp").isNotNull() &
            col("stats.symbol").isNotNull() &
            col("stats.window").isNotNull() &
            col("stats.avg_price").isNotNull() &
            col("stats.std_price").isNotNull()
        )

        # Left join price_df with stats_df based on price.timestamp falling in window
        joined_df = price_df.alias("price").join(
            stats_df,
            [
                col("price.symbol") == col("stats.symbol"),
                col("price.timestamp").isNotNull(),
                col("stats.timestamp").isNotNull(),
                col("price.timestamp") >= col("stats.timestamp") - expr("interval 1 hour"),
                col("price.timestamp").cast("long").between(
                    col("stats.timestamp").cast("long") - col("stats.window_seconds"),
                    col("stats.timestamp").cast("long") - 1
                )
            ],
            "left_outer"
        )

        # Compute Z-score
        zscore_df = joined_df.select(
            col("price.timestamp").alias("price_timestamp"),
            col("price.symbol").alias("symbol"),
            struct(
                col("stats.window"),
                when(
                    (col("stats.std_price").isNotNull()) &
                    (col("stats.std_price") != 0) &
                    (col("price.price").isNotNull()) &
                    (col("stats.avg_price").isNotNull()),
                    (col("price.price") - col("stats.avg_price")) / col("stats.std_price")
                ).otherwise(lit(None)).alias("zscore_price")
            ).alias("zscore_struct")
        )

        # Group by price_timestamp and symbol to collect Z-scores
        result_df = zscore_df.groupBy("price_timestamp", "symbol") \
            .agg(collect_list("zscore_struct").alias("zscores"))

        # Define UDF to deduplicate zscores array by window
        @udf(ArrayType(StructType([
            StructField("window", StringType(), True),
            StructField("zscore_price", DoubleType(), True)
        ])))
        def deduplicate_zscores(zscores: list) -> list:
            if not zscores:
                return []
            # Create a dictionary to keep the first zscore_price for each window
            seen_windows = {}
            for zscore in zscores:
                window = zscore["window"] if zscore and "window" in zscore else None
                zscore_price = zscore["zscore_price"] if zscore and "zscore_price" in zscore else None
                if window and window not in seen_windows:
                    seen_windows[window] = {
                        "window": window,
                        "zscore_price": zscore_price
                    }
            # Return the deduplicated list
            return list(seen_windows.values())

        # Apply deduplication to zscores array
        result_df = result_df.select(
            col("price_timestamp").alias("timestamp"),
            col("symbol"),
            deduplicate_zscores(col("zscores")).alias("zscores")
        )

        return result_df

class PipelineOrchestrator:
    """Coordinates the Z-score computation pipeline."""
    def __init__(self,
                 kafka_bootstrap_servers: str,
                 price_topic: str,
                 moving_topic: str,
                 output_topic: str,
                 checkpoint_dir: str,
                 watermark_duration: str = "5 minutes"): # Added watermark_duration
        self.spark_config = SparkConfig("ZScoreProcessor", kafka_bootstrap_servers)
        self.spark = self.spark_config.create_spark_session()
        self.kafka_reader = KafkaReader(self.spark, kafka_bootstrap_servers, watermark_duration) # Pass to reader
        self.kafka_writer = KafkaWriter(kafka_bootstrap_servers, checkpoint_dir)
        self.zscore_processor = ZScoreProcessor()
        self.price_topic = price_topic
        self.moving_topic = moving_topic
        self.intermediate_topic = output_topic + "-wins"
        self.output_topic = output_topic
        self.watermark_duration = watermark_duration  # Store it

    def run(self):
        """Runs the Z-score computation pipeline."""

        # Read price and moving streams
        price_df = self.kafka_reader.read_price_stream(self.price_topic)
        moving_df = self.kafka_reader.read_moving_stream(self.moving_topic)

        # Print the schema of the input data
        print("Price stream schema:")
        price_df.printSchema()
        print("Moving stream schema:")
        moving_df.printSchema()

        # Process moving_df to stats format and write to intermediate topic
        stats_df = self.zscore_processor.process_moving_to_stats(moving_df)
        self.kafka_writer.write_stream(stats_df, self.intermediate_topic, "zscore-wins")

        # Read stats stream
        stats_input_df = self.kafka_reader.read_intermediate_stream(self.intermediate_topic)
        print("Stats stream schema:")
        stats_input_df.printSchema()

        # Compute Z-scores
        zscore_df = self.zscore_processor.compute_zscores(price_df, stats_input_df)

        # Write results to output topic with append mode
        self.kafka_writer.write_stream(zscore_df, self.output_topic, "zscore", output_mode="append")

        # Await termination
        self.spark.streams.awaitAnyTermination()

def main():
    checkpoint_path = "file:///tmp/spark_checkpoint/btc-price-zscore"
    pipeline = PipelineOrchestrator(
        kafka_bootstrap_servers="localhost:9092",
        price_topic="btc-price",
        moving_topic="btc-price-moving",
        output_topic="btc-price-zscore",
        checkpoint_dir=checkpoint_path,
        watermark_duration="10 seconds"
    )
    pipeline.run()

if __name__ == "__main__":
    main()
