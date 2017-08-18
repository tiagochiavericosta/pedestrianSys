#!/usr/bin/python

# imports
import cv2
from multiprocessing import Process, Queue, Value
from timer import Timer
from threading import Thread
import argparse
import os, time

# global objects
frame_queue = []
result_queue = []
face_coordinates_queue = []

def parse_args():
    """
     Parse all arguments from cli
    :return:
    """
    parser = argparse.ArgumentParser(description='Pedestrian Detection Demo System')
    parser.add_argument('--device', dest='device_name', required=True,
                        choices = {"desktop": 1 , "jetson": 0},
                        default=1, help='device type for running system')
    args = parser.parse_args()
    return args




# entry
if __name__ == '__main__':
    # get args
    args = parse_args()

    quitSys = Value('i', 0)

    # queues








