# Configure Kafka Topics and Running Instructions

This section provides instructions for configuring Kafka topics and running the executable files for the BTCUSDT price analysis pipeline as per Lab 04: Spark Streaming requirements. The pipeline includes Extract, Transform (Moving Statistics and Z-score), Load, and Bonus stages.

## Configure Kafka Topics

Ensure Kafka is running on `localhost:9092` (or adjust the configuration as needed). Create the following Kafka topics using the `kafka-topics.sh` script:

```bash
bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic <topic_name>
```

Required topics:

- **Extract**:
  - `btc-price`: Stores raw price data fetched from the Binance API.
- **Transform (Moving Statistics)**:
  - **`btc-price-moving-wins`**: **Intermediate topic added by the group** to store flattened moving statistics (average and standard deviation) before aggregation.
  - `btc-price-moving`: Stores final moving statistics results.
- **Transform (Z-score)**:
  - **`btc-price-zscore-wins`**: **Intermediate topic added by the group** to store flattened statistics for Z-score computation.
  - `btc-price-zscore`: Stores final Z-score results.
- **Bonus**:
  - `btc-price-higher`: Stores time windows for price increases.
  - `btc-price-lower`: Stores time windows for price decreases.

Example command to create the `btc-price` topic:

```bash
bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic btc-price
```

Repeat for the remaining topics: `btc-price-moving-wins`, `btc-price-moving`, `btc-price-zscore-wins`, `btc-price-zscore`, `btc-price-higher`, and `btc-price-lower`.

**Note**: The topics `btc-price-moving-wins` and `btc-price-zscore-wins` are **intermediate topics** introduced by the group to facilitate data processing in the Transform stage, simplifying joins and aggregations.

## Running Instructions

The executable files are located in the `src/` directory and are run using `spark-submit`, except for the Extract stage, which uses Python. Ensure Kafka and MongoDB are running before executing the scripts.

1. **Extract (`<GroupID>.py`)**:
   - **File**: `src/Extract/<GroupID>.py`
   - **Description**: Fetches BTCUSDT price data from the Binance API and publishes it to the `btc-price` topic.
   - **Run**:
     ```bash
     python src/Extract/<GroupID>.py
     ```
   - **Note**: Ensure the Binance API (`api.binance.com/api/v3/ticker/price?symbol=BTCUSDT`) is accessible and the `btc-price` topic exists.

2. **Transform - Moving Statistics (`<GroupID>_moving.py`)**:
   - **File**: `src/Transform/<GroupID>_moving.py`
   - **Description**: Computes moving averages and standard deviations, publishing intermediate results to **`btc-price-moving-wins`** (group-added intermediate topic) and final results to `btc-price-moving`.
   - **Run**:
     ```bash
     spark-submit src/Transform/<GroupID>_moving.py
     ```
   - **Note**: The `btc-price` topic must contain data, and `btc-price-moving-wins` and `btc-price-moving` topics must exist.

3. **Transform - Z-score (`<GroupID>_zscore.py`)**:
   - **File**: `src/Transform/<GroupID>_zscore.py`
   - **Description**: Computes Z-scores using price and moving statistics, publishing intermediate results to **`btc-price-zscore-wins`** (group-added intermediate topic) and final results to `btc-price-zscore`.
   - **Run**:
     ```bash
     spark-submit src/Transform/<GroupID>_zscore.py
     ```
   - **Note**: The `btc-price` and `btc-price-moving` topics must contain data, and `btc-price-zscore-wins` and `btc-price-zscore` topics must exist.

4. **Load (`<GroupID>.py`)**:
   - **File**: `src/Load/<GroupID>.py`
   - **Description**: Reads Z-scores from `btc-price-zscore` and stores them in MongoDB collections (`btc_price_zscore_<window>`).
   - **Run**:
     ```bash
     spark-submit --packages "org.mongodb.spark:mongo-spark-connector_2.12:10.4.1,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5" src/Load/<GroupID>.py
     ```
   - **Note**: Set the `MONGO_URI` environment variable in `load.py`. The `btc-price-zscore` topic must contain data.

5. **Bonus (`<GroupID>_bonus.py`)**:
   - **File**: `src/Bonus/<GroupID>_bonus.py`
   - **Description**: Identifies time windows for price increases and decreases, publishing results to `btc-price-higher` and `btc-price-lower`.
   - **Run**:
     ```bash
     spark-submit src/Bonus/<GroupID>_bonus.py
     ```
   - **Note**: The `btc-price` topic must contain data, and `btc-price-higher` and `btc-price-lower` topics must exist.

### Execution Order

1. Run `extract.py` to populate `btc-price` with price data.
2. Run `<GroupID>_moving.py` to generate moving statistics in `btc-price-moving-wins` and `btc-price-moving`.
3. Run `<GroupID>_zscore.py` to compute Z-scores in `btc-price-zscore-wins` and `btc-price-zscore`.
4. Run `<GroupID>.py` (Load) to store Z-scores in MongoDB.
5. Run `<GroupID>_bonus.py` (optional) for Bonus processing.

### Additional Notes

- **Checkpoint Directories**: Each Spark script uses checkpoint directories at `/tmp/spark_checkpoint_...`. Delete these directories before re-running to avoid conflicts.
- **Verify Topics**: Use a Kafka consumer to check topic data:
  ```bash
  bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic <topic_name> --from-beginning
  ```
- **MongoDB**: Verify data in MongoDB collections (`btc_price_zscore_<window>`) using MongoDB Compass or shell.
