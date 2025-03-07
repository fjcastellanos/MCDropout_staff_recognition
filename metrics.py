import numpy as np
import utilsParameters
from mean_average_precision import MetricBuilder


def calculate_mAP(detections, annotations):
    #detections = [[box[0], box[1], box[2], box[3],  0, 1] for box in boxes_prediction]
    #annotations = [[boxes_gt[idx_box][0], boxes_gt[idx_box][1], boxes_gt[idx_box][2], boxes_gt[idx_box][3], 0, 0, list_idx_pages[idx_box]] for idx_box in range(len(boxes_gt))]
    
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    # Agrega las detecciones y anotaciones
    metric_fn.add(np.array(detections), np.array(annotations))

    # Calcula el mAP
    mAP = metric_fn.value(iou_thresholds=0.5)['mAP']
    
    COCO = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    print("mAP:" + str(mAP))
    print(f"COCO mAP: " + str(COCO))
    
    return mAP, COCO


"""IoU and F1"""
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list): Bounding box 1 in the format [x1, y1, x2, y2].
        box2 (list): Bounding box 2 in the format [x1, y1, x2, y2].

    Returns:
        float: IoU value.
    """
    # Get points of boxes
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    # Calculate dimensions of boxes
    w1, h1 = x12 - x11, y12 - y11
    w2, h2 = x22 - x21, y22 - y21

    # Calculate areas of both boxes
    area_box1: float = w1 * h1
    area_box2: float = w2 * h2

    # Calculate coordinates of the intersection rectangle
    x_intersection = max(x11, x21)
    y_intersection = max(y11, y21)
    w_intersection = max(0, min(x12, x22) - x_intersection)
    h_intersection = max(0, min(y12, y22) - y_intersection)

    # Calculate area of intersection
    area_intersection: float = w_intersection * h_intersection

    # Calculate area of union
    area_union: float = area_box1 + area_box2 - area_intersection

    # Calculate IoU
    return (area_intersection / area_union) if (area_union > 0.0) else 0.0

def calculate_F1(y_true, y_pred, iou_threshold: float =0.5):
    """Calculate F1 score

    Args:
        y_true (list): List of true bounding boxes
        y_pred (list): List of predicted bounding boxes
        iou_threshold (float): IoU Threshold to math prediction with true. Defaults to 0.5.

    Returns:
        float: F1 score
        list: list of matched IoUs
    """
    num_preds = len(y_pred)
    num_true  = len(y_true)

    if num_preds == 0 and num_true == 0:
        return 1, [], 0,0,0

    true_positives  = 0.0
    false_positives = num_preds
    false_negatives = num_true

    matrix_of_iou = np.asarray([[calculate_iou(box_pred, box_true) for box_true in y_true] for box_pred in y_pred])
    matched_ious  = []

    if utilsParameters.DEBUG_FLAG:
      print(f'Matrix of IoU:\n{matrix_of_iou}')

    while matrix_of_iou.size > 0:
        # Find the maximum IoU in the matrix
        current_max_IoU = np.max(matrix_of_iou)
        if current_max_IoU < iou_threshold:
            break

        # Add the current max IoU to matched IoUs
        matched_ious.append(current_max_IoU)

        # This is a match, update TP, FP, FN
        true_positives  += 1
        false_negatives -= 1
        false_positives -= 1

        # Get the indices of the maximum IoU
        max_index = np.unravel_index(matrix_of_iou.argmax(), matrix_of_iou.shape)

        # Remove the matched bounding boxes (row and column) from the IoU matrix
        matrix_of_iou = np.delete(matrix_of_iou, max_index[1], axis=1)  # Remove column
        matrix_of_iou = np.delete(matrix_of_iou, max_index[0], axis=0)  # Remove row

    # Number of predictions that haven't been matched (false predictions)
    
    f1score, precision, recall = getF1_from_TP_FP_FN(true_positives, false_positives, false_negatives)
    
    return f1score, matched_ious, true_positives, false_positives, false_negatives


def getF1_from_TP_FP_FN(true_positives, false_positives, false_negatives):
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.
    recall    = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives)  > 0 else 0.

    if utilsParameters.DEBUG_FLAG:
      print(f'Precision: {precision} \t Recall: {recall}')

    # print(f'{precision} -- {recall}')
    f1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1score, precision, recall