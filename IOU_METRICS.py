from collections import namedtuple
import numpy as np
import cv2
import matplotlib.pyplot as plt

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


examples = [

    Detection("C:/Users/27820/Desktop/Campus Stuff/MASTERS/cfd/CFD Version 2.0.3/CFD 2.0.3 "
              "Images/BM-019/CFD-BM-019-002-N.jpg", [1058, 1121, 1358, 1260], [1015, 1100, 1393, 1289])



]


for detection in examples:
    # load the image
    image = cv2.imread(detection.image_path)
    # draw the ground-truth bounding box along with the predicted
    # bounding box
    cv2.rectangle(image, tuple(detection.gt[:2]),
                  tuple(detection.gt[2:]), (0, 255, 0), 2)
    cv2.rectangle(image, tuple(detection.pred[:2]),
                  tuple(detection.pred[2:]), (0, 0, 255), 2)
    # # compute the intersection over union and display it
    iou = bb_intersection_over_union(detection.gt, detection.pred)
    cv2.putText(image, "IoU: {:.4f}".format(iou), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0), 2)
    cv2.putText(image, "Green: Ground Truth", (100, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)
    cv2.putText(image, "Red: Predicted", (100, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
    print("{}: {:.4f}".format(detection.image_path, iou))
    # show the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.axis("off")
