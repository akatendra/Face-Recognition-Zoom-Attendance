# Обработка одного скриншота с именем zoom.jpg. В отдельный поток запускается
# функция face_encodings_func()
from threading import Thread
import cv2 as cv
import numpy as np
import face_recognition as fr
import copy

from time import time
from tqdm import tqdm


class FaceEncodingsThread(Thread):
    # constructor
    def __init__(self):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.value = None

    # function executed in a new thread
    def run(self):
        self.value = face_encodings_func(zoom_image, face_locations)


def face_encodings_func(img, face_locations):
    face_encodings = fr.face_encodings(img, face_locations)
    # Find all facial features in all the faces in the image
    face_landmarks_list = fr.face_landmarks(img)
    return face_encodings, face_landmarks_list


##################### PATHES #####################

path = 'class_got'
ZOOMPATH = 'zoom_got_landmarks'

##################### Read recognition data #####################
start_time = time()
known_face_names = np.load(
    f'{path}/names.npy')  # LIST CONTAINING ALL THE CORRESPONDING CLASS Names
known_face_encodings = np.load(
    f'{path}/face_encodings.npy')  # LIST CONTAINING ALL THE CORRESPONDING KNOWN FACE
print('Recognition data received', time() - start_time, 'sec')
##################### ZOOM #####################


zoom_image = fr.load_image_file(f'{ZOOMPATH}/zoom.jpg')
zoom_image = cv.cvtColor(zoom_image, cv.COLOR_BGR2RGB)
start_time = time()
face_locations = fr.face_locations(zoom_image)

face_count = len(face_locations)

print(f"There are {face_count} students in this class", time() - start_time,
      'sec')

# face_encodings = fr.face_encodings(zoom_image, face_locations) # The most longest process
start_time = time()
encoding_thread = FaceEncodingsThread()
encoding_thread.start()

while encoding_thread.is_alive():
    print(f'Processing... ({time() - start_time})', end='')
    print('\r', end='')

encoding_thread.join()
print('Face encoding of ZOOM-image complete', time() - start_time, 'sec')
face_encodings = encoding_thread.value[0]

for (top, right, bottom, left), face_encoding in zip(face_locations,
                                                     face_encodings):
    # See if the face is a match for the known face(s)
    matches = fr.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = fr.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
        name_len = len(name)
    frame_width = right - left
    name_plate_width = name_len * 10  # 10 is width of one letter approximately.
    if frame_width < name_plate_width:
        d_left = int((name_plate_width - frame_width) / 2)
    else:
        d_left = 1

    # Draw a box around the face using the CV2
    cv.rectangle(zoom_image, (left, top), (right, bottom), (255, 0, 255),
                 2)

    # Draw a label with a name below the face
    cv.rectangle(zoom_image, (left - d_left, bottom),
                 (right + d_left, bottom + 35),
                 (255, 0, 255), cv.FILLED)
    cv.putText(zoom_image, name, (left - d_left + 5, bottom + 20),
               cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
# Draw landmarks on image
face_landmarks_list = encoding_thread.value[1]
# Green color in BGR
color = (0, 255, 0)
# Copy zoom_image for landmarks
zoom_image_lines = copy.copy(zoom_image)
zoom_image_circles = copy.copy(zoom_image)

# Line thickness of 9 px
thickness = 1
for face_landmarks in face_landmarks_list:

    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print(
            f'The {facial_feature} in this face has the following points: {face_landmarks[facial_feature]}')

    # Let's trace out each facial feature in the image with a line!
    for facial_feature in face_landmarks.keys():
        start_point = None
        for point in face_landmarks[facial_feature]:
            end_point = point
            if start_point is not None:
                cv.line(zoom_image_lines, start_point, end_point, color,
                        thickness)
            start_point = end_point


    # Let's trace out each facial feature in the image with a point as circle!
    for facial_feature in face_landmarks.keys():
        for point in face_landmarks[facial_feature]:
            # Draw a green circle with zero radius and -1 for filled circle
            cv.circle(zoom_image_circles, point, radius=2,
                      color=(0, 255, 0), thickness=-1)


    # # Let's trace out each facial feature in the image with a pixel point!
    # for facial_feature in face_landmarks.keys():
    #     for point in face_landmarks[facial_feature]:
    #         # Draw a green pixel
    #         zoom_image_pixels[point[0], point[1]] = [0, 255, 0]

cv.imshow('Zoom with landmarks in lines', zoom_image_lines)
cv.imshow('Zoom with landmarks in circles', zoom_image_circles)
cv.imshow('Zoom', zoom_image)
cv.waitKey(0)
