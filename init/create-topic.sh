#!/bin/bash
set -e

TOPIC="device_data.streams"

echo "Waiting for Kafka..."
sleep 5

/opt/kafka/bin/kafka-topics.sh --create \
  --topic "$TOPIC" \
  --bootstrap-server kafka:9092 \
  --partitions 1 \
  --replication-factor 1 \
  || echo "Topic already exists."

echo "Topic ready: $TOPIC"
