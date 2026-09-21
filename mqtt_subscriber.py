import paho.mqtt.client as mqtt
import json
import psycopg2
from datetime import datetime

# --- Configuration ---
# Must match the publisher's configuration
BROKER = "test.mosquitto.org" 
PORT = 1883
TOPIC = "vit-ind-03/sensor/telemetry"

# Database credentials (ensure these match your schema.sql environment)
DB_CREDENTIALS = {
    'dbname': 'vit_ind_03_db',
    'user': 'postgres',
    'password': 'your_password',
    'host': 'localhost',
    'port': 5432
}

def insert_telemetry_to_db(data):
    """
    Insert the received MQTT JSON payload into PostgreSQL TBL_SENSOR_READING_LOG.
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CREDENTIALS)
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO TBL_SENSOR_READING_LOG 
            (Sensor_ID, Timestamp, Signal_Value, Unit, Sampling_Rate, Data_Quality) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            data.get('sensor_id'),
            data.get('timestamp'),
            data.get('signal_value'),
            data.get('unit'),
            data.get('sampling_rate'),
            data.get('data_quality')
        ))
        
        conn.commit()
        cursor.close()
        print(f"✅ DB Insert Success: Saved reading {data.get('signal_value')} {data.get('unit')}")
        
    except psycopg2.Error as e:
        print(f"❌ Database insertion failed: {e}")
        # Note: In production, you might want to log this to a file or dead-letter queue
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        if conn:
            conn.close()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✅ Connected to MQTT Broker! Subscribing to '{TOPIC}'...")
        client.subscribe(TOPIC)
    else:
        print(f"❌ Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """
    Callback function triggered every time a message is published to the subscribed topic.
    """
    try:
        payload_str = msg.payload.decode('utf-8')
        data = json.loads(payload_str)
        
        print(f"📥 Received MQTT message at {data.get('timestamp')}")
        
        # Write the real-time telemetry to the database
        # Make sure to uncomment this once your DB is running!
        
        # insert_telemetry_to_db(data)
        
    except json.JSONDecodeError:
        print(f"❌ Error: Received malformed JSON payload: {msg.payload}")
    except Exception as e:
        print(f"❌ Error processing message: {e}")

# Initialize MQTT Client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "Cloud_Ingestion_Service_001")
client.on_connect = on_connect
client.on_message = on_message

print(f"Starting Ingestion Service. Connecting to {BROKER}:{PORT}...")
client.connect(BROKER, PORT, 60)

# Blocking call that processes network traffic, dispatches callbacks, and handles reconnecting.
try:
    print("Listening for incoming telemetry... (Press Ctrl+C to stop)")
    client.loop_forever()
except KeyboardInterrupt:
    print("\nIngestion service stopped by user.")
    client.disconnect()
