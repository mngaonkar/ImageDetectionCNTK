import os, sys
import numpy as np
from cntk import load_model
import utils.od_utils as od
from utils.config_helpers import merge_configs
from utils.annotations.annotations_helper import parse_class_map_file

def get_configuration(detector_name):
    # load configs for detector, base network and data set
    if detector_name == "FastRCNN":
        from FastRCNN.FastRCNN_config import cfg as detector_cfg
    elif detector_name == "FasterRCNN":
        from FasterRCNN.FasterRCNN_config import cfg as detector_cfg
    else:
        print('Unknown detector: {}'.format(detector_name))

    # for VGG16 base model use:         from utils.configs.VGG16_config import cfg as network_cfg
    # for AlexNet base model use:       from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    # for Pascal VOC 2007 data set use: from utils.configs.Pascal_config import cfg as dataset_cfg
    # for the Grocery data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg
    from utils.configs.Grocery_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg, {'DETECTOR': detector_name}])

def evaluate_model(eval_model, detector_name):
    cfg = get_configuration(detector_name)
    cfg['NUM_CHANNELS'] = 3
    print("Map file = ", cfg["DATA"].CLASS_MAP_FILE)
    cfg["DATA"].CLASSES = parse_class_map_file(os.path.join("Steer_Bad_Relevant_output", cfg["DATA"].CLASS_MAP_FILE))
    cfg["DATA"].NUM_CLASSES = len(cfg["DATA"].CLASSES)

      # detect objects in single image
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"Steer_Bad_Relevant_output/testImages/Steer_Bad_Front_Zoom (269).jpg")
    regressed_rois, cls_probs = od.evaluate_single_image(eval_model, img_path, cfg)
    bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, cfg)

    fg_boxes = np.where(labels > 0)
    print("#bboxes: before nms: {}, after nms: {}, foreground: {}".format(len(regressed_rois), len(bboxes), len(fg_boxes[0])))
    for i in fg_boxes[0]: print("{:<12} (label: {:<2}), score: {:.3f}, box: {}".format(
                                cfg["DATA"].CLASSES[labels[i]], labels[i], scores[i], [int(v) for v in bboxes[i]]))

    od.visualize_results(img_path, bboxes, labels, scores, cfg, store_to_path="Steer_Bad_Relevant_output/output.jpg")

if __name__ == '__main__':
    eval_model = load_model("FasterRCNN/Output/faster_rcnn_eval_AlexNet_4stage.model")
    evaluate_model(eval_model, 'FasterRCNN')

