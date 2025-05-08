from kafka import KafkaProducer
import json
import time
import random
from datetime import datetime, timezone

# Kafka cấu hình
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Danh sách các khung thời gian (windows)
windows = ["30s", "1m", "5m", "15m", "30m", "1h"]

# Vòng lặp gửi dữ liệu giả lập
while True:
    timestamp = datetime.now(timezone.utc).isoformat()
    symbol = "BTC"
    
    # Tạo danh sách z-score cho mỗi cửa sổ
    zscores = [
        {"window": w, "zscore_price": round(random.uniform(-2.0, 2.0), 4)}
        for w in windows
    ]

    message = {
        "timestamp": timestamp,
        "symbol": symbol,
        "windows": zscores
    }

    producer.send("btc-price-zscore", value=message)
    print("Sent:", message)

    time.sleep(1)  # gửi mỗi giây một bản tin (có thể thay đổi)
