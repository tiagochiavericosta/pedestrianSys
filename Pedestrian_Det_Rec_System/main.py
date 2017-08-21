#!/usr/bin/env python2.7

# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
from camera import CameraSrc
from detector import faceDetector
import multiprocessing
from multiprocessing import Process, Queue, Value, cpu_count, get_logger, Array
from timer import Timer
from threading import Thread
import argparse
import os, time
import logging
import sys



# global objects
sav_frame = None
sav_result = []

def parse_args():
    """
     Parse all arguments from cli
    :return:
    """
    parser = argparse.ArgumentParser(description='Pedestrian Detection Demo System')
    parser.add_argument('--device', dest='device', required=True,
                        choices = {"0": "Jetson", "1": "Desktop"},
                        default=1, help='device type for running system. 0: Jetson, 1: Desktop')
    parser.add_argument('--fullsc', dest='fullscreen', default=False,
                        action='store_true', help='Enable Full screen mode')
    args = parser.parse_args()
    return args

# detection process
def detection(quit, det):

    while quit.value == 0:
        # get frame from queue
        frame = frame_queue.get()

        # perform detection
        alignedFaces, face_locs = det.detect_faces(frame=frame)

        if alignedFaces is not None:
            # add to aligned faces queue
            face_aligned_queue.put(alignedFaces)
            face_coordinates_queue.put(face_locs)

# stream process
def stream(quit, det):
    global sav_frame
    global sav_result

    camera_src = None
    if args.device == 0:  # jetson
        camera_src = CameraSrc().get_cam_src()
    else:  # desktop
        camera_src = 0

    camera = cv2.VideoCapture(camera_src)
    assert camera.isOpened()

    if args.fullscreen:
        cv2.namedWindow(args.device, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(args.device,
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    face_locs = None
    alignedFace = None

    while True:
        _, frame = camera.read()
        sav_frame = frame.copy()

        # add frames to queue
        frame_queue.put(sav_frame)

        # display detection results
        face_locs = face_coordinates_queue.get()
        alignedFace = face_aligned_queue.get()

        if face_locs is not None:
            print(len(alignedFace), face_locs)

        det.display(frame=frame, face_locations=face_locs)

        cv2.imshow(args.device, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit.value = 1
            break

    #producer.join()
    camera.release()
    cv2.destroyAllWindows()


# Main Process
if __name__ == '__main__':
    # get args
    args = parse_args()

    quitSys = Value('i', 0)

    # queues
    frame_queue = Queue()
    result_queue = Queue()
    face_coordinates_queue = Queue()
    face_aligned_queue = Queue()
    message = Array('c', 'Now Loading...' + (' ' * 20))

    # init
    multiprocessing.log_to_stderr()
    logger = get_logger()
    logger.setLevel(logging.INFO)

    # detector
    det = faceDetector()

    # streaming producer process
    stream_process = Process(
        target=stream, args=(quitSys, det)
    )
    stream_process.start()

    # detection consumer process
    detection_process = Process(
        target=detection, args=(quitSys, det)
    )
    detection_process.start()

    t = Timer()
    # consumer pro-type
    while quitSys.value == 0:
        t.tic()


        # run recognition

        t.toc()
        t.clear()

    # exit all processes
    print('-- Exiting --')
    frame_queue.close()
    result_queue.close()
    face_coordinates_queue.close()
    face_aligned_queue.close()

    frame_queue.join_thread()
    result_queue.join_thread()
    face_coordinates_queue.join_thread()
    face_aligned_queue.join_thread()

    stream_process.terminate()
    detection_process.terminate()
    stream_process.join()
    detection_process.join()
    sys.exit(0)
    # end








