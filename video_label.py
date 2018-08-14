import argparse
import csv
import os
import cv2
import tqdm
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def set_gpu_options():
    return
    gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0.3)
    config_proto = tf.ConfigProto(
            gpu_options=gpu_options)

def load_detection_graph(CKPT_PATH):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(CKPT_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def get_df(all_rows):
    df = pd.DataFrame(all_rows,
                      columns=['frame', 'cname', 'conf', 'xmin', 'ymin', 'xmax', 'ymax'])
    f32 = ['conf', 'xmin', 'ymin', 'xmax', 'ymax']
    for f in f32:
        df[f] = df[f].astype('float32')
    df['frame'] = df['frame'].astype('int32')
    df['cname'] = df['cname'].astype('int8')
    df = df.sort_values(by=['frame', 'cname', 'conf'], ascending=[True, True, False])
    print(df)
    return df

def label_video(detection_graph, category_index, cap, feather_fname,
                nb_frames=100000000, start_frame=0):
    all_rows = []
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config_proto) as sess:
            for i in tqdm.trange(nb_frames):
                frame = start_frame + i
                ret, im = cap.read()
                if not ret:
                    break
                # BGR -> RGB
                frame = frame[...,::-1]
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
                for j, box in enumerate(boxes):
                    if scores[j] < 0.3:
                        continue
                    ymin, xmin, ymax, xmax = boxes[j]
                    # object_name = category_index[classes[j]]['name']
                    object_name = classes[j] # saves hella space
                    row = [i, object_name, scores[j], xmin, ymin, xmax, ymax]
                    all_rows.append(row)

    df = get_df(all_rows)
    feather.write_dataframe(df, feather_fname)
    cap.release()

def get_category_index(LABEL_PATH, NB_CLASSES=90):
    label_map = label_map_util.load_labelmap(LABEL_PATH)
    categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NB_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--label_path', required=True,
                        default='./research/object_detection/data/mscoco_label_map.pbtxt')
    parser.add_argument('--start_frame', default=0, type=int)
    parser.add_argument('--nb_frames', default=10000000, type=int)
    parser.add_argument('--video_fname', required=True, type=str)
    parser.add_argument('--index_fname', type=str)
    parser.add_argument('--feather_fname', required=True, type=str)
    args = parser.parse_args()

    detection_graph = load_detection_graph(args.ckpt_path)
    category_index = get_category_index(args.label_path)


    USE_SWAG = True
    if USE_SWAG:
        cap = swag.VideoCapture(args.video_fname, args.index_fname)
    else:
        cap = cv2.VideoCapture(args.video_fname)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    label_video(detection_graph, category_index, cap, args.feather_fname,
                args.nb_frames, args.start_frame)

if __name__ == '__main__':
    main()
