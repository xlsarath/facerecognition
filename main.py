import cv2
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
from twilio.rest import Client 
import os
import smtplib
from datetime import datetime
import argparse
import sys
import re
import json
import time
import numpy as np
import pandas as pd
import os,glob

TIMEOUT = 10 #10 seconds
count = 0
def main(args):
    mode = args.mode
    if(mode == "camera"):
        camera_recog()
    elif mode == "input":
        create_manual_data();
    else:
        raise ValueError("Unimplemented mode")

def is_FileCheck(path):
    #print(path)
    for ll in os.listdir(os.path.dirname(path)):
            #print(ll)
            if re.sub(r"(\*+)|(.csv)","",(os.path.basename(path))) in ll:
                bol = True
    return bol

def concat_files(filePath):
    #print(filePath)
    #print(is_FileCheck(filePath))
    try:
        if(is_FileCheck(filePath)):
                frames =[]
                for l in  os.listdir(os.path.dirname(filePath)):
                    if(re.sub(r"(\*+)|(.csv)","",(os.path.basename(filePath))) in l):
                        if(re.sub(r"(\*+)|(.csv)","",(os.path.basename(filePath))) == 'Results'):
                            df1=pd.read_csv(os.path.dirname(filePath)+"\\"+l, index_col=None, skiprows  = 1)
                        else:
                            df1=pd.read_csv(os.path.dirname(filePath)+"\\"+l, index_col=None)
                        frames.append(df1)  
                return df1
        else:
            print("No files present in dir for case")
            #exit
    except:
        #print("Error encountered:",sys.exc_info())   

def camera_recog():
    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture(0); #get input from webcam
    detect_time = time.time()
    while True:
        _,frame = vs.read();
        rects, landmarks = face_detect.detect_face(frame,80);#min face size is set to 80x80
        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160,frame,landmarks[:,i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else: 
                print("Align face failed") #log        
        if(len(aligns) > 0):
            features_arr = extract_feature.get_features(aligns)
            recog_data = findPeople(features_arr,positions)
            for (i,rect) in enumerate(rects):
                cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0)) #draw bounding box for the face
                cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)


        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
def findPeople(features_arr, positions, thres = 0.6, percent_thres = 70):
    f = open('./facerec_128D.txt','r')
    data_set = json.loads(f.read());
    returnRes = [];
    for (i,features_128D) in enumerate(features_arr):
        result = "Unknown";
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]];
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance;
                    result = person;
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
        returnRes.append((result,percentage))
        checkUnknown(result)
    return returnRes    

# importing twilio 

# Your Account Sid and Auth Token from twilio.com / console 

def create_manual_data():
    vs = cv2.VideoCapture(0); #get input from webcam
    print("Please input new user ID:")
    new_name = input(); #ez python input()
    f = open('./facerec_128D.txt','r');
    data_set = json.loads(f.read());
    person_imgs = {"Left" : [], "Right": [], "Center": []};
    person_features = {"Left" : [], "Right": [], "Center": []};
    while True:
        _, frame = vs.read();
        rects, landmarks = face_detect.detect_face(frame, 80);  # min face size is set to 80x80
        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(160,frame,landmarks[:,i]);
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                person_imgs[pos].append(aligned_frame)
                cv2.imshow("Captured face", aligned_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    for pos in person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
    data_set[new_name] = person_features;
    f = open('./facerec_128D.txt', 'w');
    f.write(json.dumps(data_set))


import math

import cv2
import numpy as np


class AlignCustom(object):
    def __init__(self):
        pass
    
    def getPos(self, points):
        if abs(points[0] - points[2]) / abs(points[1] - points[2]) > 2:
            return "Right";
        elif abs(points[1] - points[2]) / abs(points[0] - points[2]) > 2:
            return "Left";
        return "Center"

    def list2colmatrix(self, pts_list):
        assert len(pts_list) > 0
        colMat = []
        for i in range(len(pts_list)):
            colMat.append(pts_list[i][0])
            colMat.append(pts_list[i][1])
        colMat = np.matrix(colMat).transpose()
        return colMat

    def find_tfrom_between_shapes(self, from_shape, to_shape):
        assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

        sigma_from = 0.0
        sigma_to = 0.0
        cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        # compute the mean and cov
        from_shape_points = from_shape.reshape(int(from_shape.shape[0] / 2), 2)
        to_shape_points = to_shape.reshape(int(to_shape.shape[0] / 2), 2)
        mean_from = from_shape_points.mean(axis=0)
        mean_to = to_shape_points.mean(axis=0)

        for i in range(from_shape_points.shape[0]):
            temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
            sigma_from += temp_dis * temp_dis
            temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
            sigma_to += temp_dis * temp_dis
            cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)

        sigma_from = sigma_from / to_shape_points.shape[0]
        sigma_to = sigma_to / to_shape_points.shape[0]
        cov = cov / to_shape_points.shape[0]

        # compute the affine matrix
        s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        u, d, vt = np.linalg.svd(cov)

        if np.linalg.det(cov) < 0:
            if d[1] < d[0]:
                s[1, 1] = -1
            else:
                s[0, 0] = -1
        r = u * s * vt
        c = 1.0
        if sigma_from != 0:
            c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

        tran_b = mean_to.transpose() - c * r * mean_from.transpose()
        tran_m = c * r

        return tran_m, tran_b

    def align(self, desired_size, img, landmarks, padding=0.1):
        shape = []
        for k in range(int(len(landmarks) / 2)):
            shape.append(landmarks[k])
            shape.append(landmarks[k + 5])

        if padding > 0:
            padding = padding
        else:
            padding = 0
        mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
        mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]

        from_points = []
        to_points = []

        for i in range(int(len(shape) / 2)):
            x = (padding + mean_face_shape_x[i]) / (2 * padding + 1) * desired_size
            y = (padding + mean_face_shape_y[i]) / (2 * padding + 1) * desired_size
            to_points.append([x, y])
            from_points.append([shape[2 * i], shape[2 * i + 1]])

        from_mat = self.list2colmatrix(from_points)
        to_mat = self.list2colmatrix(to_points)

        tran_m, tran_b = self.find_tfrom_between_shapes(from_mat, to_mat)

        probe_vec = np.matrix([1.0, 0.0]).transpose()
        probe_vec = tran_m * probe_vec

        scale = np.linalg.norm(probe_vec)
        angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])

        from_center = [(shape[0] + shape[2]) / 2.0, (shape[1] + shape[3]) / 2.0]
        to_center = [0, 0]
        to_center[1] = desired_size * 0.4
        to_center[0] = desired_size * 0.5

        ex = to_center[0] - from_center[0]
        ey = to_center[1] - from_center[1]

        rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]), -1 * angle, scale)

        rot_mat[0][2] += ex
        rot_mat[1][2] += ey

        chips = cv2.warpAffine(img, rot_mat, (desired_size, desired_size))
        return chips, self.getPos(landmarks)



if __name__ == '__main__':
    process_kill = open("C:\\Users\psma018\\Desktop\\FaceRec-master\\faceRecog.bat", "w+")
    process_kill.write("taskkill /F /PID "+str(os.getpid()))
    process_kill.close()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
    args = parser.parse_args(sys.argv[1:]);
    FRGraph = FaceRecGraph();
    MTCNNGraph = FaceRecGraph();
    aligner = AlignCustom();
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2); #scale_factor, rescales image for faster detection
    main(args);
