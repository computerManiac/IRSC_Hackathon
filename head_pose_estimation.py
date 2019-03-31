import cv2
import dlib
import numpy as np
from imutils import face_utils
from img_utils import *
from scipy.spatial import distance as dist
from threading import Thread
from math import log
import time

face_landmark_path = './shape_predictor.dat'

def calDist(currSpeed, rTime, deacc):
    brakingDist = -np.square(currSpeed)/(2*deacc)
    rDist = rTime*currSpeed

    return rDist + brakingDist

def collision(rTime, total_distance, currSpeed, deacc):
    myDist = calDist(currSpeed, rTime, deacc)
    if myDist > total_distance:
        myDist = myDist - (myDist-total_distance)
    score = 1/(log(1+(total_distance-myDist)) + 1e-5)

    return score

def calFatigue(time, warningC, warningD, total_distance):
    t = np.exp(warningC*warningD)
    return (t/time)

def reactTime(fatigue):
    return 2.3*(1+fatigue)


def EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A+B)/(2.0*C)

    return ear

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def main():
    MAX_FRAMES = 5
    count = 0
    EYE_AR = 0.25
    EYE_AR_FRAMES = 5
    counter_ear = 0
    lstart, lend = 42, 48
    rstart, rend = 36, 42

    warningD = 1.1
    warningC = 1.1

    #simulation variables
    total_distance = int(input('Enter distance: '))
    speed = int(input('Enter speed: '))
    deacc = 0.78*9.8

    start = time.time()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)

    while cap.isOpened():
        elapsed = time.time() - start
        if elapsed >= 1:
            total_distance -= speed
            print('Distance: ' + str(total_distance))
        ret, frame = cap.read()
        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lstart:lend]
                rightEye = shape[rstart:rend]

                leftEAR = EAR(leftEye)
                rightEAR = EAR(rightEye)
                ear = (leftEAR + rightEAR)/2.0

                lHull = cv2.convexHull(leftEye)
                rHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [lHull], -1, (0,255,0), 1)
                cv2.drawContours(frame, [rHull], -1, (0,255,0), 1)

                reprojectdst, euler_angle = get_head_pose(shape)

                if ear < EYE_AR:
                    counter_ear += 1

                    if counter_ear >= EYE_AR_FRAMES:
                        cv2.putText(frame, "ALERT!!", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                        warningD *= np.exp(1000/total_distance)

                else:
                    counter_ear = 0

                x,y,w,h = rect2bb(face_rects[0])
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

                cv2.putText(frame, "Face", (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(frame, "EAR {}".format(ear), (x-10,y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                if abs(euler_angle[0,0]) >= 20 or abs(euler_angle[1,0]) >= 20 or abs(euler_angle[2,0]) >= 20:
                    count += 1
                    if count > MAX_FRAMES:
                        cv2.putText(frame, "ALERT!!", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                        warningC *= np.exp(1000/total_distance)

                elif abs(euler_angle[0,0]) <= 15 and abs(euler_angle[1,0]) <= 15 and abs(euler_angle[2,0]) <= 15:
                    count = 0

                for start, end in line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)

                fatigue = calFatigue(elapsed, warningC, warningD, total_distance)
                print("Fatigue: " + str(fatigue) + " Time: " + str(elapsed))
                rTime = reactTime(fatigue)
                score = collision(rTime, total_distance, speed, deacc)
                cv2.putText(frame, f"Score {score}", (100,430), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
