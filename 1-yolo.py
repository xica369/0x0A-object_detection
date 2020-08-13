#!/usr/bin/env python3

"""Initialize Yolo"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """class Yolo"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        - model_path is the path to where a Darknet Keras model is stored
        - classes_path is the path to where the list of class names used
          for the Darknet model, listed in order of index, can be found
        - class_t is a float representing the box score threshold for
          the initial filtering step
        - nms_t is a float representing the IOU threshold for non-max
          suppression
        - anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
          containing all of the anchor boxes:
            * outputs is the number of outputs (predictions)
            * made by the Darknet model
            * anchor_boxes is the number of anchor boxes used for each
              prediction
            * 2 => [anchor_box_width, anchor_box_height]"""

        # Public instance attributes
        self.model = K.models.load_model(model_path)

        self.class_names = []

        with open(classes_path, "r") as file:
            for line in file:
                class_name = line.strip()
                self.class_names.append(class_name)

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        public method

        - outputs is a list of numpy.ndarrays containing the predictions from
          the Darknet model for a single image:
          Each output will have the shape:
          (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            * grid_height & grid_width => the height and width of the grid used
              for the output
            * anchor_boxes => the number of anchor boxes used
            * 4 => (t_x, t_y, t_w, t_h)
            * 1 => box_confidence
            * classes => class probabilities for all classes
        - image_size is a numpy.ndarray containing the image’s original size
          [image_height, image_width]

        Returns a tuple of (boxes, box_confidences, box_class_probs):
        - boxes: a list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, 4) containing the processed
          boundary boxes for each output, respectively:
          * 4 => (x1, y1, x2, y2)
          * (x1, y1, x2, y2) should represent the boundary box relative to
            original image
        - box_confidences: a list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, 1) containing the box
          confidences for each output, respectively
        - box_class_probs: a list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, classes) containing the box’s
          class probabilities for each output, respectively
        """

        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height = image_size[0]
        image_width = image_size[1]

        input_height = self.model.input.shape[2].value
        input_width = self.model.input.shape[1].value

        for outp in outputs:

            # create list with np.ndarray (grid_h, grid_w, anchor_boxes, 4)
            out = outp[..., :4]
            boxes.append(out)

            # calculate confidences for each output
            box_confidence = 1 / (1 + np.exp(-(outp[..., 4:5])))
            box_confidences.append(box_confidence)

            # calcule class probabilities for each output
            box_class_prob = 1 / (1 + np.exp(-(outp[..., 5:])))
            box_class_probs.append(box_class_prob)

        for iter, box in enumerate(boxes):
            grid_hight = box.shape[0]
            grid_width = box.shape[1]
            anchors_box = box.shape[2]

            # create matrix Cy
            matrix_cy = np.arange(grid_hight).reshape(1, grid_hight)
            matrix_cy = np.repeat(matrix_cy, grid_width, axis=0).T
            matrix_cy = np.repeat(matrix_cy[:, :, np.newaxis],
                                  anchors_box,
                                  axis=2)

            # create matrix Cx
            matrix_cx = np.arange(grid_width).reshape(1, grid_width)
            matrix_cx = np.repeat(matrix_cx, grid_hight, axis=0)
            matrix_cx = np.repeat(matrix_cx[:, :, np.newaxis],
                                  anchors_box,
                                  axis=2)

            # calculate sigmoid to tx and ty
            box[..., :2] = 1 / (1 + np.exp(-(box[..., :2])))

            # calculate bx = sigmoid(tx) + Cx
            box[..., 0] += matrix_cx

            # calculate by = sigmoid(ty) + Cy
            box[..., 1] += matrix_cy

            anchor_width = self.anchors[iter, :, 0]
            anchor_hight = self.anchors[iter, :, 1]

            # calculate e(tw) and e(th)
            box[..., 2:] = np.exp(box[..., 2:])

            # calculate bw = anchor_width * e(tw)
            box[..., 2] *= anchor_width

            # calculate bh = anchor_hight * e(th)
            box[..., 3] *= anchor_hight

            # adjust scale
            box[..., 0] *= image_width / grid_width
            box[..., 1] *= image_height / grid_hight
            box[..., 2] *= image_width / input_width
            box[..., 3] *= image_height / input_height

            # calculate x1 = tx - bw / 2
            box[..., 0] -= box[..., 2] / 2

            # calculate y1 = ty - bh / 2
            box[..., 1] -= box[..., 3] / 2

            # calculate x2 = x1 + bw
            box[..., 2] += box[..., 0]

            # calculate y2 = y1 + bh
            box[..., 3] += box[..., 1]

        return (boxes, box_confidences, box_class_probs)
