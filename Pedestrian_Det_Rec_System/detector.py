"""
    This is a module for Face Detection

    - by philipchicco
"""

# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import time
import os
from darkflow.net.build import TFNet
from camera import CameraSrc

start = time.time()
img_dims = 96


# custom enum class: No enum support in python 2.7
class SuperEnum(object):
    class __metaclass__(type):
        def __iter__(self):
            def __iter__(self):
                """Return a generator for enum values.

                When iterating the enum, only return class members whose
                value areq identical to the key.
                """
                return (item for item in self.__dict__ if item == self.__dict__[item])


# models : make this accessible to both modules
class ModelsDir(SuperEnum):
    models = "models"
    labels = "labels"
    yolo_face_cfg = "yolo-face.cfg"
    yolo_face_weights = "yolo-face_final.weights"
    face_label_file = "labels.txt"
    face_model_cfg = os.path.join(models,yolo_face_cfg)
    face_model_wgts = os.path.join(models,yolo_face_weights)
    face_labels = os.path.join(labels, face_label_file)


# face detection module constructs
class faceDetector(object):

    # pass model paths etc as arguments
    def __init__(self):
        """
        Constructor for detection module
        """
        try:
            model_path = ModelsDir.face_model_cfg
            model_weights = ModelsDir.face_model_wgts
            threshold = 0.1
            gpu = 0.7
            labels = ModelsDir.face_labels

            print("--Loading ")
            print(model_path, model_weights)

            options = {
                "model" : model_path,
                "load" : model_weights,
                "threshold" : float(threshold),
                "gpu" : float(gpu),
                "labels" : labels,
            }

            # load model
            self._tfnet = TFNet(options)

        except Exception:  # exception must be caught at call
            raise Exception("Error initialising detection module")

    def modelInstance(self):
        return self._tfnet

    def resize_frame(self, frame):
        """
         - Resize frame of video to 1/4 size for
         faster face recognition processing
        :param frame: input video frame
        :return: resized image frame
        """
        return cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    def resize(self, frame):
        h, w, c = frame.shape
        if h > 480 and w > 640:
            return True
        return False

    # requires revision to use tensorflow framework
    def detect_faces(self, frame):
        """
        - Detect faces in a frame.
        :param frame: image frame
        :return: locations of faces in a list
        """
        alignedFaces = None
        bb = None

        start = time.time()

        # check frame
        if frame is None:
            raise Exception("Invalid frame: frame is None")

        if self.resize(frame):
            frame = self.resize_frame(frame)

        # color conversion
        rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # bounding boxes

        # verbose
        # print("Alignment tooks {} seconds.".format(time.time() - start))

        # send to recognition Module to align
        return alignedFaces, self.face_locations(img_shape=frame.shape, bb=bb)

    # needs to be revised
    def display(self, frame, face_locations):
        """
        - Display results on screen with bboxes
        :param frame: window frame
        :return: window with resulting predictions on faces
        """
        # Display the results
        scale = 1
        if self.resize:
            scale = 4

        if not len(face_locations) == 0:  # nothing detected
            for (top, right, bottom, left) in face_locations:
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top * scale
                right * scale
                bottom * scale
                left * scale

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                # else

    def display_detected(self, frame, face_locs, people, confidence):
        """
        - Display ROI's of detected faces with labels
        :param frame:
        :param face_locs:
        :param people : people in image classified
        :param confidence : recognition confidence
        :return:
        """

        if not len(face_locs) == 0:  # nothing detected
            for (top, right, bottom, left), name, conf in zip(face_locs, people, confidence):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top
                right
                bottom
                left

                # string
                conf_4f = "%.3f" % conf
                peop_conf = "{} {}%".format(name, float(conf_4f) * 100)

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.rectangle(frame, (left, top + 20), (right, top), (0, 0, 255), cv2.FILLED)

                font = cv2.FONT_HERSHEY_DUPLEX  # color
                # cv2.putText(frame, peop_conf , (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, peop_conf, (left, top + 15), font, 0.5, (255, 255, 255), 1)
        pass

    def _rect_to_css(self, rect):
        """
        Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

        :param rect: a dlib 'rect' object
        :return: a plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return rect.top(), rect.right(), rect.bottom(), rect.left()

    def _css_to_rect(self, css):
        """
        Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :return: a dlib `rect` object
        """
        return dlib.rectangle(css[3], css[0], css[1], css[2])

    def _trim_css_to_bounds(self, css, image_shape):
        """
        Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :param image_shape: numpy shape of the image array
        :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

    def face_locations(self, img_shape, bb):
        """
        Returns an array of bounding boxes of human faces in a image

        :param img: An image (as a numpy array)
        :param bb: bounding boxes regions
        :return: A list of tuples of found face locations in css (top, right, bottom, left) order
        """
        return [self._trim_css_to_bounds(self._rect_to_css(face), img_shape) for face in bb]

