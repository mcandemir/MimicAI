import numpy as np
import tensorflow as tf
import cv2

from object_detection.utils import visualization_utils as viz_utils

# setting min confidence threshold
MIN_CONF_THRESH=.6


model= tf.saved_model.load('models/my_model/saved_model')
# PATH_TO_LABELS = 'annotations/label_map.pbtxt'
category_index = {1:{'id': 1, 'name': 'VG Panel with Missing Teeth'},2:{'id': 2, 'name': 'Erosion'},3:{'id': 3,
'name': 'Receptor'},4:{'id': 4, 'name': 'VG Panel'}}
# NUM_CLASSES = 4
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
#                                                             use_display_name=True)
# category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array of shape (height, width, channels), where channels=3 for RGB to feed into tensorflow graph.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))



def process_image(image_path):
    # read image and preprocess
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    #print(detections['detection_classes'])
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=MIN_CONF_THRESH,
      agnostic_mode=False)
    cv2.imwrite(image_path, cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR))
