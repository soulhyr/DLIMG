import numpy as np

def predict(confs, locs, default_boxes, NUM_CLASSES, conf_thresh=0.5):
    confs = softmax(confs, axis=-1).astype(np.float32)
    classes = np.argmax(confs, axis=-1)
    scores = np.max(confs, axis=-1).astype(np.float32)
    boxes = decode(default_boxes, locs).astype(np.float32)

    batch_boxes = []
    batch_classes = []
    batch_scores = []

    for confs_each_img, boxes_each_img in zip(confs, boxes):
        out_boxes = []
        out_labels = []
        out_scores = []

        for c in range(1, NUM_CLASSES):
            cls_scores = confs_each_img[:, c]
            score_idx = cls_scores > conf_thresh
            cls_boxes = boxes_each_img[score_idx]
            cls_scores = cls_scores[score_idx]

            nms_idx = compute_nms(cls_boxes, cls_scores, 0.45, 200)
            cls_boxes = np.take(cls_boxes, nms_idx, axis=0)
            cls_scores = np.take(cls_scores, nms_idx, axis=0)
            cls_labels = [c] * cls_boxes.shape[0]

            out_boxes.append(cls_boxes)
            out_labels.extend(cls_labels)
            out_scores.append(cls_scores)
        out_boxes = np.concatenate(out_boxes, axis=0)
        out_scores = np.concatenate(out_scores, axis=0)

        batch_boxes.append(np.clip(out_boxes, 0.0, 1.0))
        batch_classes.append(np.array(out_labels))
        batch_scores.append(out_scores)
    return batch_boxes, batch_classes, batch_scores

def xyxy_to_xywh(boxes):
    return np.stack([boxes[:,0],boxes[:,1],boxes[:,2]-boxes[:,0], boxes[:,3]-boxes[:,1]]).T

def xywh_to_xyxy(boxes):
    return np.stack([boxes[:,0],boxes[:,1],boxes[:,2]+boxes[:,0], boxes[:,3]+boxes[:,1]]).T

def absolute_coord(boxes, height, width):
    abs_coord = np.stack([boxes[:,0]*width, boxes[:,1]*height, boxes[:,2]*width, boxes[:,3]*height]).T
    return abs_coord.astype(np.int32)

def relative_coord(boxes, height, width):
    rel_coord = np.stack([boxes[:,0]/width, boxes[:,1]/height, boxes[:,2]/width, boxes[:,3]/height]).T
    return rel_coord.astype(np.float32)

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis, keepdims=True))
    return e_x / np.sum(e_x, axis, keepdims=True) # only difference

def transform_center_to_corner(boxes):
    """ Transform boxes of format (cx, cy, w, h)
        to format (xmin, ymin, xmax, ymax)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    corner_box = np.concatenate([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2], axis=-1)

    return corner_box

def decode(default_boxes, locs, variance=[0.1, 0.2]):
    """ Decode regression values back to coordinates
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        locs: tensor (batch_size, num_default, 4)
              of format (cx, cy, w, h)
        variance: variance for center point and size
    Returns:
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    locs = np.concatenate([
        locs[..., :2] * variance[0] *
        default_boxes[:, 2:] + default_boxes[:, :2],
        np.exp(locs[..., 2:] * variance[1]) * default_boxes[:, 2:]], axis=-1)

    boxes = transform_center_to_corner(locs)

    return boxes

def compute_area(top_left, bot_right):
    """ Compute area given top_left and bottom_right coordinates
    Args:
        top_left: tensor (num_boxes, 2)
        bot_right: tensor (num_boxes, 2)
    Returns:
        area: tensor (num_boxes,)
    """
    # top_left: N x 2
    # bot_right: N x 2
    hw = np.clip(bot_right - top_left, 0.0, 1.0)
    area = hw[..., 0] * hw[..., 1]

    return area

def compute_iou(boxes_a, boxes_b):
    """ Compute overlap between boxes_a and boxes_b
    Args:
        boxes_a: tensor (num_boxes_a, 4)
        boxes_b: tensor (num_boxes_b, 4)
    Returns:
        overlap: tensor (num_boxes_a, num_boxes_b)
    """
    # boxes_a => num_boxes_a, 1, 4
    boxes_a = np.expand_dims(boxes_a, 1)

    # boxes_b => 1, num_boxes_b, 4
    boxes_b = np.expand_dims(boxes_b, 0)
    top_left = np.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bot_right = np.minimum(boxes_a[..., 2:], boxes_b[..., 2:])

    overlap_area = compute_area(top_left, bot_right)
    area_a = compute_area(boxes_a[..., :2], boxes_a[..., 2:])
    area_b = compute_area(boxes_b[..., :2], boxes_b[..., 2:])

    overlap = overlap_area / (area_a + area_b - overlap_area)

    return overlap

def compute_nms(boxes, scores, nms_threshold, limit=200):
    """ Perform Non Maximum Suppression algorithm
        to eliminate boxes with high overlap

    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
        scores: tensor (num_boxes,)
        nms_threshold: NMS threshold
        limit: maximum number of boxes to keep

    Returns:
        idx: indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    selected = [0]
    idx = np.argsort(-scores)
    idx = idx[:limit]
    boxes = np.take(boxes, idx, axis=0)
    iou = compute_iou(boxes, boxes)
    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold
        # iou[:, ~next_indices] = 1.0
        iou = np.where(
            np.expand_dims(np.logical_not(next_indices), 0),
            np.ones_like(iou, dtype=np.float32),
            iou)

        if not np.any(next_indices):
            break
        selected.append(np.argsort(-(next_indices.astype(np.int32)), kind='mergesort')[0])

    return np.take(idx, selected, axis=0)
