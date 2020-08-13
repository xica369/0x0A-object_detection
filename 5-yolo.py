#!/usr/bin/env python3

"""Initialize Yolo"""

import tensorflow.keras as K
import numpy as np
import glob
import cv2


class Yolo:
    """class Yolo"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path is the path to where a Darknet Keras model is stored
        classes_path is the path to where the list of class names used
        for the Darknet model, listed in order of index, can be found
        class_t is a float representing the box score threshold for
        the initial filtering step
        nms_t is a float representing the IOU threshold for non-max suppression
        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes:
          outputs is the number of outputs (predictions)
          made by the Darknet model
          anchor_boxes is the number of anchor boxes used for each prediction
          2 => [anchor_box_width, anchor_box_height]"""

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
        """Process Outputs"""

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter Boxes"""

        class_t = self.class_t
        filtered_boxes = []
        box_classes = []
        box_scores = []

        scores = []
        for box_confid, box_class_prob in zip(box_confidences,
                                              box_class_probs):
            scores.append(box_confid * box_class_prob)

        for score in scores:

            box_score = score.max(axis=-1)
            box_score = box_score.flatten()
            box_scores.append(box_score)

            box_class = np.argmax(score, axis=-1)
            box_class = box_class.flatten()
            box_classes.append(box_class)

        box_scores = np.concatenate(box_scores, axis=-1)
        box_classes = np.concatenate(box_classes, axis=-1)

        for box in boxes:
            filtered_boxes.append(box.reshape(-1, 4))

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)

        filtering_mask = np.where(box_scores >= self.class_t)

        filtered_boxes = filtered_boxes[filtering_mask]
        box_classes = box_classes[filtering_mask]
        box_scores = box_scores[filtering_mask]

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        method that applies Non-max Suppression
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for classes in set(box_classes):
            index = np.where(box_classes == classes)

            filtered = filtered_boxes[index]
            scores = box_scores[index]
            classe = box_classes[index]

            x1 = filtered[:, 0]
            x2 = filtered[:, 2]
            y1 = filtered[:, 1]
            y2 = filtered[:, 3]

            keep = []
            area = (x2 - x1) * (y2 - y1)
            index_list = np.flip(scores.argsort(), axis=0)

            while len(index_list) > 0:
                pos1 = index_list[0]
                pos2 = index_list[1:]
                keep.append(pos1)

                xx1 = np.maximum(x1[pos1], x1[pos2])
                yy1 = np.maximum(y1[pos1], y1[pos2])
                xx2 = np.minimum(x2[pos1], x2[pos2])
                yy2 = np.minimum(y2[pos1], y2[pos2])

                height = np.maximum(0.0, yy2 - yy1)
                width = np.maximum(0.0, xx2 - xx1)

                intersection = (width * height)
                union = area[pos1] + area[pos2] - intersection
                iou = intersection / union
                below_threshold = np.where(iou <= self.nms_t)[0]
                index_list = index_list[below_threshold + 1]

            keep = np.array(keep)

            box_predictions.append(filtered[keep])
            predicted_box_classes.append(classe[keep])
            predicted_box_scores.append(scores[keep])

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """function that load images

        folder_path: a string representing the path to the folder holding all
        the images to load
        Returns a tuple of (images, image_paths):
        images: a list of images as numpy.ndarrays
        image_paths: a list of paths to the individual images in images"""

        images = []

        image_paths = glob.glob(folder_path + "/*")

        for image in image_paths:
            img_read = cv2.imread(image)
            images.append(img_read)

        return (images, image_paths)

    def preprocess_images(self, images):
        """Preprocess images"""

        input_width = self.model.input.shape[1].value
        input_height = self.model.input.shape[2].value

        resize = (input_width, input_height)
        image_shapes = []
        pimages = []

        for image in images:
            shape = image.shape[:2]
            image_shapes.append(shape)

            image_resize = cv2.resize(image,
                                      resize,
                                      interpolation=cv2.INTER_CUBIC)

            image_rescaled = image_resize/255

            pimages.append(image_rescaled)

        pimages = np.stack(pimages, axis=0)
        image_shapes = np.stack(image_shapes, axis=0)

        return (pimages, image_shapes)
