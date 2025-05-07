import time
import json
import requests
from kafka import KafkaProducer
from datetime import datetime, timezone

class BTCPriceProducer:
    def __init__(self, kafka_bootstrap_servers, topic, fetch_interval_ms=100):
        self.topic = topic
        self.fetch_interval = fetch_interval_ms / 1000.0  # convert ms to seconds

        # Khởi tạo Kafka Producer
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def fetch_price(self):
        """
        Fetch BTC price from Binance API.
        """
        response = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT')
        response.raise_for_status()  # Raise exception nếu lỗi HTTP
        data = response.json()

        message = {
            "symbol": data["symbol"],
            "price": float(data["price"]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return message

    def publish_price(self, message):
        """
        Publish the fetched price to Kafka topic.
        """
        self.producer.send(self.topic, value=message)
        print(f"Sent: {message}")

    def run(self):
        """
        Main loop: fetch price and publish periodically.
        """
        print(f"Starting BTCPriceProducer: pushing to topic '{self.topic}' every {self.fetch_interval}s.")
        while True:
            try:
                message = self.fetch_price()
                self.publish_price(message)
                time.sleep(self.fetch_interval)

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

# Nếu chạy trực tiếp file này, thì chạy main
if __name__ == "__main__":
    producer = BTCPriceProducer(
        kafka_bootstrap_servers='localhost:9092',
        topic='btc-price',
        fetch_interval_ms=100
    )
    producer.run()