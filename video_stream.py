import argparse
import collections
import common
import gstreamer
import numpy as np
import os
import re
import svgwrite
import time
from app_secrets import BIGQUERY_PROJECT, BIGQUERY_TABLE
import threading
from pytz import timezone
from email_util import send_email
import cv2
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from google.cloud import bigquery
from send_pubsub import send_to_pubsub
# from flask_server import frame_queue

# Your model config
MODEL_DIR = '/home/mendel/CoralCode/model'
MODEL = 'safety_model2_v2_edgetpu.tflite'
LABEL = 'safety_objection_label.txt'

# Initialize the bigquery
client = bigquery.Client(project=BIGQUERY_PROJECT)
table_id = BIGQUERY_TABLE

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

#Global hashmap to keep unique id for each object
id_mapping = {}
id_detection_times = {} 
email_sent = {} 


def send_email_async(filename):
    try:
        send_email(filename)
        print('Email sent successfully.')
    except Exception as e:
        print('Failed to send email. Reason: %s' % e)

def process_email(input_tensor):
    # Get all memory blocks in the buffer
    mem = input_tensor.get_all_memory() #print(f"Memory blocks: {mem}")

    # Map the memory blocks and get the data
    success, mapinfo = mem.map(Gst.MapFlags.READ)
    if not success:
        raise ValueError("Could not map memory blocks")

    # print(f"Success in mapping memory blocks: {success}")
    # print(f"MapInfo: {mapinfo}")

    # Create a numpy array from the data
    data = np.frombuffer(mapinfo.data, dtype=np.uint8)  # Replace np.uint8 with the correct type #print(f"Numpy data shape: {data.shape}, type: {data.dtype}")

    # Now data is a numpy array which you can use in your computations:
    frame = (data * 255).astype(np.uint8) #print(f"Frame shape: {frame.shape}, type: {frame.dtype}")

    # Save the frame
    frame = data.reshape(320, 320, 3)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite('detected_frame.jpg', frame)

    # Use a separate thread to send the email
    email_thread = threading.Thread(target=send_email_async, args=('detected_frame.jpg',))
    email_thread.start()

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
        lines = (p.match(line).groups() for line in f.readlines())
        return {int(num): text.strip() for num, text in lines}


def shadow_text(dwg, x, y, text, font_size=20):
    dwg.add(dwg.text(text, insert=(x+1, y+1), fill='black', font_size=font_size))
    dwg.add(dwg.text(text, insert=(x, y), fill='white', font_size=font_size))


def generate_svg(src_size, inference_size, inference_box, objs, labels, text_lines, trdata, trackerFlag):
    dwg = svgwrite.Drawing('', size=src_size)
    src_w, src_h = src_size
    inf_w, inf_h = inference_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h

    for y, line in enumerate(text_lines, start=1):
        shadow_text(dwg, 10, y*20, line)
    if trackerFlag and (np.array(trdata)).size:
        for td in trdata:
            x0, y0, x1, y1, trackID = td[0].item(), td[1].item(
            ), td[2].item(), td[3].item(), td[4].item()
            overlap = 0
            for ob in objs:
                dx0, dy0, dx1, dy1 = ob.bbox.xmin.item(), ob.bbox.ymin.item(
                ), ob.bbox.xmax.item(), ob.bbox.ymax.item()
                area = (min(dx1, x1)-max(dx0, x0))*(min(dy1, y1)-max(dy0, y0))
                if (area > overlap):
                    overlap = area
                    obj = ob

            # Relative coordinates.
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            # Absolute coordinates, input tensor space.
            x, y, w, h = int(x * inf_w), int(y *
                                             inf_h), int(w * inf_w), int(h * inf_h)
            # Subtract boxing offset.
            x, y = x - box_x, y - box_y
            # Scale to source coordinate space.
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = '{}% {} ID:{}'.format(
                percent, labels.get(obj.id, obj.id), int(trackID))
            shadow_text(dwg, x, y - 5, label)
            # Choose the color of the bounding box based on the object ID.
            if obj.id == 0:
                box_color = 'green'
            elif obj.id == 1:
                box_color = 'red'
            # else:
            #     box_color = 'blue'  # Default to blue 

            dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                            fill='none', stroke=box_color, stroke_width='2'))
        return dwg.tostring()
    
    
    else:
        for obj in objs:
            x0, y0, x1, y1 = list(obj.bbox)
            # Relative coordinates.
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            # Absolute coordinates, input tensor space.
            x, y, w, h = int(x * inf_w), int(y *
                                             inf_h), int(w * inf_w), int(h * inf_h)
            # Subtract boxing offset.
            x, y = x - box_x, y - box_y
            # Scale to source coordinate space.
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
            shadow_text(dwg, x, y - 5, label)
            # Choose the color of the bounding box based on the object ID.
            if obj.id == 0:
                box_color = 'green'
            elif obj.id == 1:
                box_color = 'red'
            # else:
            #     box_color = 'blue'  # Default to blue 

            dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                            fill='none', stroke=box_color, stroke_width='2'))
    return dwg.tostring()


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()


def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    category_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(category_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))
    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]


def main():
    default_model_dir = MODEL_DIR
    default_model = MODEL
    default_labels = LABEL
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='classifier score threshold')
    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video0')
    parser.add_argument('--videofmt', help='Input video format.',
                        default='raw',
                        choices=['raw', 'h264', 'jpeg'])
    parser.add_argument('--tracker', help='Name of the Object Tracker To be used.',
                        default='sort',
                        choices=[None, 'sort'])
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    w, h, _ = common.input_image_size(interpreter)
    inference_size = (w, h)
    # Average fps over last 30 frames.
    fps_counter = common.avg_fps_counter(30)

    def user_callback(input_tensor, src_size, inference_box, mot_tracker):
        nonlocal fps_counter
        start_time = time.monotonic()
        common.set_input(interpreter, input_tensor)
        interpreter.invoke()
        # For larger input image sizes, use the edgetpu.classification.engine for better performance
        objs = get_output(interpreter, args.threshold, args.top_k)
        end_time = time.monotonic()
        detections = []  # np.array([])
        for n in range(0, len(objs)):
            #print(f'obj id {objs[n].id}')
            element = []  # np.array([])
            element.append(objs[n].bbox.xmin)
            element.append(objs[n].bbox.ymin)
            element.append(objs[n].bbox.xmax)
            element.append(objs[n].bbox.ymax)
            element.append(objs[n].score)  # print('element= ',element)
            detections.append(element)  # print('dets: ',dets)

        # convert to numpy array #      print('npdets: ',dets)
        detections = np.array(detections)
        trdata = []
        trackerFlag = False
        global id_mapping
        global id_detection_times 
        global email_sent

        if detections.any():
            if mot_tracker != None:
                trdata = mot_tracker.update(detections) #print(f"trdata: {trdata}") 
                if trdata.any():
                    for i, unique_track in enumerate(trdata):
                        tracker_id = unique_track[4] 
                        object_id = objs[i].id 
                        print(f"Tracker ID: {tracker_id}, Object ID: {object_id}")  # print tracker and object IDs
                        # If the tracker id is not already in id_mapping, it's a new id
                        if tracker_id not in id_mapping:
                            id_mapping[tracker_id] = object_id
                            id_detection_times[tracker_id] = time.time() 
                            email_sent[tracker_id] = False
                            # New id detected, send data to BigQuery
                            # send_to_bigquery(objs)
                            send_to_pubsub(objs)
                        # print(id_detection_times)
                        # If object id is 1 and it's been more than 3 seconds since first detected, and email has not been sent yet
                        if object_id == 1 and tracker_id in id_detection_times:
                            time_diff = time.time() - id_detection_times[tracker_id] #print(f' the time difference is {time_diff}')
                            if time_diff > 3 and not email_sent[tracker_id]:
                                print("Sending email...")  # print a message before sending the email
                                process_email(input_tensor)
                                email_sent[tracker_id] = True 
                            #print(f"Actual Object ID: {id_mapping[tracker_id]}, BBox Coords: {unique_track[:5]}")

        text_lines = [
            'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
            'FPS: {} fps'.format(round(next(fps_counter))), 
            'Number of objects detected: {}'.format(len(objs)),]
        if len(objs) != 0:
            return generate_svg(src_size, inference_size, inference_box, objs, labels, text_lines, trdata, trackerFlag)
        else:
            dwg = svgwrite.Drawing('', size=src_size)  # Create an empty SVG image
            for y, line in enumerate(text_lines, start=1):
                shadow_text(dwg, 10, y*20, line)  # Write the text lines to the empty SVG image
            svg = dwg.tostring()  # Convert the SVG drawing to a string
        return svg
    
    result = gstreamer.run_pipeline(user_callback,
                                    src_size=(640, 480),
                                    appsink_size=inference_size,
                                    trackerName=args.tracker,
                                    videosrc=args.videosrc,
                                    videofmt=args.videofmt)


if __name__ == '__main__':
    main()