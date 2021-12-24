import numpy as np
import polyiou


def obbox2corners(rbox):
    cx, cy, w, h, alpha = np.split(rbox, 5, 1)
    cos_a_half = np.cos(alpha) * 0.5
    sin_a_half = np.sin(alpha) * 0.5
    w_x = cos_a_half * w
    w_y = sin_a_half * w
    h_x = -sin_a_half * h
    h_y = cos_a_half * h
    return np.concatenate([cx + w_x + h_x, cy + w_y + h_y,
                           cx + w_x - h_x, cy + w_y - h_y,
                           cx - w_x - h_x, cy - w_y - h_y,
                           cx - w_x + h_x, cy - w_y + h_y], axis=1)


def py_cpu_nms_poly_fast(dets, iou_thr):
    # TODO: check the type numpy()
    dets = dets.astype(dtype=np.float64)
    if dets.shape[0] == 0:
        keep = np.zeros([0], dtype=np.long)
    else:
        obbs = dets[:, 0:-1]
        # pdb.set_trace()
        x1 = np.min(obbs[:, 0::2], axis=1)
        y1 = np.min(obbs[:, 1::2], axis=1)
        x2 = np.max(obbs[:, 0::2], axis=1)
        y2 = np.max(obbs[:, 1::2], axis=1)
        scores = dets[:, 8]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        polys = []
        for i in range(len(dets)):
            tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                               dets[i][2], dets[i][3],
                                               dets[i][4], dets[i][5],
                                               dets[i][6], dets[i][7]])
            polys.append(tm_polygon)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            ovr = []
            i = order[0]
            keep.append(i)
            # if order.size == 0:
            #     break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # w = np.maximum(0.0, xx2 - xx1 + 1)
            # h = np.maximum(0.0, yy2 - yy1 + 1)
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            hbb_inter = w * h
            hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
            # h_keep_inds = np.where(hbb_ovr == 0)[0]
            h_inds = np.where(hbb_ovr > 0)[0]
            tmp_order = order[h_inds + 1]
            for j in range(tmp_order.size):
                iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
                hbb_ovr[h_inds[j]] = iou

            inds = np.where(hbb_ovr <= iou_thr)[0]
            order = order[inds + 1]

    return dets[keep, :], np.array(keep)


def multiclass_poly_nms_rbbox(multi_rbboxes,
                              multi_scores,
                              score_thr,
                              nms_thr,
                              max_num=-1,
                              score_factors=None):
    """
    NMS for multi-class bboxes.
    :param multi_rbboxes: [N, 8]
    :param multi_scores: [N, class]
    :param score_thr:
    :param nms_cfg:
    :param max_num:
    :param score_factors:
    :return:
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    for i in range(num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not np.any(cls_inds):
            continue
        if multi_rbboxes.shape[1] == 5:
            _bboxes = multi_rbboxes[cls_inds, :]
            _bboxes = obbox2corners(_bboxes)
        elif multi_rbboxes.shape[1] == 8:
            _bboxes = multi_rbboxes[cls_inds, :]

        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = np.concatenate([_bboxes, _scores[:, None]], axis=1)
        # TODO: figure out the nms_cfg
        cls_dets, _ = py_cpu_nms_poly_fast(cls_dets, nms_thr)

        cls_labels = np.full((cls_dets.shape[0], ), i, dtype=np.int32)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = np.concatenate(bboxes, axis=0)
        labels = np.concatenate(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = np.zeros((0, 9))
        labels = np.zeros((0,), dtype=np.int32)

    return bboxes, labels

def multiclass_poly_nms_rbbox_patches(multi_rbboxes,
                                      multi_labels,
                                      num_classes,
                                      nms_thr):
    """
    NMS for multi-class bboxes.
    :param multi_rbboxes: [N, 9]
    :param multi_scores: [N, class]
    :param score_thr:
    :param nms_cfg:
    :param max_num:
    :param score_factors:
    :return:
    """
    bboxes, labels = [], []
    for i in range(num_classes):
        cls_inds = multi_labels == i
        if not np.any(cls_inds):
            continue
        if multi_rbboxes.shape[1] == 9:
            cls_dets_ = multi_rbboxes[cls_inds, :]
        else:
            raise ValueError("Unsupport input shape.")

        cls_dets, _ = py_cpu_nms_poly_fast(cls_dets_, nms_thr)

        cls_labels = np.full((cls_dets.shape[0], ), i, dtype=np.int32)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = np.concatenate(bboxes, axis=0)
        labels = np.concatenate(labels)
    else:
        bboxes = np.zeros((0, 9))
        labels = np.zeros((0,), dtype=np.int32)

    return bboxes, labels