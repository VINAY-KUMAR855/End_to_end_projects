import torch
from IOU import intersection_over_union


def non_max_suppression(
        bboxes,
        prob_threshold,
        iou_threshold,
        box_format = "corners"
    ):
    '''
    Parameters:
        bboxes (list): List of lists containing all boxes in the form [[class, prob_ofThat_class, x1,y1,x2,y2],[],..]
        prob_threshold (float): threshold to remove prdicted boxes(independent of IOU)
        iou_threshold (float): threshold where predicted boxes are correct
    '''
    assert type(bboxes)==list
    # remove the boxes that is less than the threshold
    bboxes = [box for box in bboxes if box[1]>prob_threshold]
    # sort the boxes like highest threshold are 1st
    bboxes = sorted(bboxes, lambda x:x[1], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        choosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != choosen_box[0] # if the box class is not the choosen_box class then we just keep it.
            or intersection_over_union(
                torch.tensor(choosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]
        bboxes_after_nms.append(choosen_box)
    return bboxes_after_nms

