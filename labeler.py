import argparse
import csv
import os
import cv2
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

parser = argparse.ArgumentParser()
parser.add_argument('--video_in', required=True)
parser.add_argument('--csv_out', required=True)
parser.add_argument('--start_frame', type=int, required=True)
args = parser.parse_args()

NB_FRAMES = 300000
NB_CLASSES = 90
LABEL_PATH = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')
CKPT_PATH = 'object_detection/ssd_inception_v2_coco_11_06_2017/frozen_inference_graph.pb'


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(CKPT_PATH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(LABEL_PATH)
categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NB_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

writer = csv.writer(open(args.csv_out, 'wb'))
# cap = cv2.VideoCapture('/lfs/1/ddkang/noscope/data/videos/taipei-long.mp4')
cap = cv2.VideoCapture(args.video_in)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        for i in xrange(args.start_frame, args.start_frame + NB_FRAMES):
            ret, frame = cap.read()
            if not ret:
                break
            tf_frame = np.expand_dims(frame, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: tf_frame})
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)
            '''vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            out_video.append(image)'''
            # FIXME
            if i % 500 == 0:
                print i
            for j, box in enumerate(boxes):
                if scores[j] < 0.3:
                    continue
                ymin, xmin, ymax, xmax = boxes[j]
                object_name = category_index[classes[j]]['name']
                row = [i, object_name, scores[j], xmin, ymin, xmax, ymax]
                writer.writerow(row)
