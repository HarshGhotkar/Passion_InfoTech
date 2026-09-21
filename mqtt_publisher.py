import paho.mqtt.client as mqtt
import time
import json
import random
import uuid
from datetime import datetime

# --- Configuration ---
# Using a public test broker for ease of development. 
# For production, install Eclipse Mosquitto locally and use 'localhost'
BROKER = "test.mosquitto.org" 
PORT = 1883
TOPIC = "vit-ind-03/sensor/telemetry"

def generate_simulated_telemetry(sensor_id):
    """
    Simulate real-time numeric sensor readings.
    While the audio processor handles heavy .wav files, this simulates 
    the continuous telemetry stream (like noise levels or temperature) 
    from the edge device.
    """
    return {
        "sensor_id": sensor_id,
        "timestamp": datetime.utcnow().isoformat(),
        # Simulating a continuous dB reading from the acoustic sensor
        "signal_value": round(random.uniform(40.0, 85.0), 2), 
        "unit": "dB",
        "sampling_rate": 1.0, # 1 reading per second
        "data_quality": round(random.uniform(0.95, 1.0), 3)
    }

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Successfully connected to MQTT Broker!")
    else:
        print(f"❌ Failed to connect, return code {rc}")

# Initialize MQTT Client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "IoT_Sensor_Edge_Sim_001")
client.on_connect = on_connect

print(f"Connecting to MQTT Broker at {BROKER}:{PORT}...")
client.connect(BROKER, PORT)
client.loop_start()

# A mock Sensor_ID that should eventually correspond to a record in TBL_SENSOR_MASTER
MOCK_SENSOR_ID = str(uuid.uuid4())

print(f"Starting simulated telemetry stream for Sensor ID: {MOCK_SENSOR_ID}")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        # Generate and publish payload
        payload = generate_simulated_telemetry(MOCK_SENSOR_ID)
        client.publish(TOPIC, json.dumps(payload))
        
        print(f"📡 Published: {payload['signal_value']} {payload['unit']} at {payload['timestamp']}")
        
        # Publish every 2 seconds
        time.sleep(2) 
except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
finally:
    client.loop_stop()
    client.disconnect()
