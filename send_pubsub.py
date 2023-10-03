import threading
from datetime import datetime
import pytz 
from app_secrets import TEST_AVRO_SCHEMA, AVRO_SCHEMA, TOPIC_ID, BIGQUERY_PROJECT
from avro.io import BinaryEncoder, DatumWriter
import avro.schema as schema
import io
import json
from google.api_core.exceptions import NotFound
from google.cloud.pubsub import PublisherClient
from google.pubsub_v1.types import Encoding
import requests

# Globals 
avro_schema = schema.parse(open(AVRO_SCHEMA, "rb").read())

def datetime_to_unix_microseconds(dt):
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.UTC)
    return int((dt - epoch).total_seconds() * 1e6)

def send_to_pubsub(objs):
    threading.Thread(target=send_to_pubsub_thread, args=(objs,)).start()

def send_to_pubsub_thread(objs):
    publisher_client = PublisherClient()
    topic_path = publisher_client.topic_path(BIGQUERY_PROJECT, TOPIC_ID)

    for obj in objs:
        id_value = obj.id
        score_value = float(obj.score)
        bbox = f"BBox(xmin={obj.bbox.xmin}, ymin={obj.bbox.ymin}, xmax={obj.bbox.xmax}, ymax={obj.bbox.ymax})"
        results = build_results(id_value, score_value, bbox)
        publish_results_to_topic(publisher_client, topic_path, results)

def build_results(id_value, score_value, bbox):
    response = requests.get('http://ip-api.com/json/')
    geodata = response.json()
    latitude = geodata['lat']
    longitude = geodata['lon']
    utc_time = datetime.now(pytz.UTC)
    results = {
        "timestamp": datetime_to_unix_microseconds(utc_time),
        "inference": {"id": int(id_value), "score": float(score_value), "bbox": str(bbox)},
        "latitude": float(latitude),
        "longitude": float(longitude),
    }
    return results

def publish_results_to_topic(publisher_client, topic_path, results):
    bout = io.BytesIO()
    writer = DatumWriter(avro_schema)

    try:
        # Get the topic encoding type.
        topic = publisher_client.get_topic(request={"topic": topic_path})
        encoding = topic.schema_settings.encoding

        # Encode the data according to the message serialization type.
        if encoding == Encoding.BINARY:
            encoder = BinaryEncoder(bout)
            writer.write(results, encoder)
            data = bout.getvalue()
        elif encoding == Encoding.JSON:
            data_str = json.dumps(results)
            data = data_str.encode("utf-8")
        else:
            print(f"No encoding specified in {topic_path}. Abort.")
            return

        future = publisher_client.publish(topic_path, data)
        print(f"Published message ID: {future.result()}")

    except NotFound:
        print(f"{topic_path} not found.")
