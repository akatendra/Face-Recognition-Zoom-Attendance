from xlsx import *

import cv2 as cv
import numpy as np
import face_recognition as fr

from time import time


def face_recognition_func():
    file_path = f'{zoom_path}/{img}'
    print(f'file: {file_path}')
    encoded_image = fr.load_image_file(file_path)
    encoded_image = cv.cvtColor(encoded_image, cv.COLOR_BGR2RGB)
    start_time_f = time()
    face_locations = fr.face_locations(encoded_image)
    print(
        f'face_locations: {face_locations}')
    face_count = len(face_locations)
    print(
        f'There are {face_count} students in this class',
        time() - start_time_f,
        'sec')
    start_time_f = time()
    face_encodings = fr.face_encodings(encoded_image,
                                       face_locations)
    print()
    print(
        f'Face encoding of ZOOM-image complete!',
        time() - start_time_f, 'sec')
    attendance_names = []
    for (top, right, bottom, left), face_encoding in zip(
            face_locations,
            face_encodings):
        # See if the face is a match for the known face(s)
        matches = fr.compare_faces(known_face_encodings,
                                   face_encoding)
        print(f'Matches: {matches}')
        name = 'Unknown'

        # Use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(known_face_encodings,
                                          face_encoding)
        print(f'Face distances: {face_distances}')
        best_match_index = np.argmin(face_distances)
        print(f'Best match index: {best_match_index}')
        name_len = len(name)  # Name length of word UNKNOWN in symbols
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            name_len = len(name)  # Name length in symbols
        print(f'Name: {name}')
        attendance_names.append(name)
        frame_width = right - left
        name_plate_width = name_len * 10  # 10 is width of one letter approximately.
        if frame_width < name_plate_width:
            d_left = int((name_plate_width - frame_width) / 2)
        else:
            d_left = 1

        # Draw a box around the face using the CV2
        cv.rectangle(encoded_image, (left, top), (right, bottom),
                     (255, 0, 255),
                     2)

        # Draw a label with a name below the face
        cv.rectangle(encoded_image, (left - d_left, bottom),
                     (right + d_left, bottom + 35),
                     (255, 0, 255), cv.FILLED)
        cv.putText(encoded_image, name,
                   (left - d_left + 5, bottom + 20),
                   cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    return attendance_names, encoded_image, face_locations, face_encodings


##################### PROGRAM MAIN BODY #####################

##################### PATH'S #####################

path = 'class_got'
zoom_path = 'zoom_got'
# path = 'class_beetroot'
# zoom_path = 'zoom_beetroot'

##################### Read recognition data #####################
start_time = time()
# List containing all the corresponding class names
known_face_names = np.load(f'{path}/names.npy')
# List containing all the corresponding known face
known_face_encodings = np.load(f'{path}/face_encodings.npy')
print('Recognition data received', time() - start_time, 'sec')

##################### Get list of screenshots #####################

file_list = os.listdir(zoom_path)  # List of files in target directory.
img_list = {}
# Iterate directory
for file in file_list:
    # check only screenshot image files
    if file.startswith('zoom_'):
        _, scr_date, scr_time = file.split('_')
        img_list[file] = [scr_date, scr_time]
img_list_len = len(img_list)
print('Обнаружено скриншотов:', img_list_len)
print('Скриншоты:', img_list)

xlsx_data = xlsx_file_read(path)  # Read XLSX-file with attendance data
print(f'xlsx_data: {xlsx_data}')

for img in img_list:
    print('img', img)
    att_date = img_list[img][0]  # Get screenshot created date
    recognised_data = face_recognition_func()
    cv.imshow(f'Screenshot #{img}', recognised_data[1])
    # Write attendance data to a dictionary
    attendance_names = recognised_data[0]  # Get names of recognised people

    for name in xlsx_data:
        att_dates = {}
        if name in attendance_names:
            if att_date not in xlsx_data[name]:
                att_dates[att_date] = 1
            else:
                print(f'Имени {name} нет в xlsx-файле!')
        else:
            if att_date not in xlsx_data[name]:
                att_dates[att_date] = 0
            else:
                print(f'Имени {name} нет в xlsx-файле!')
        xlsx_data[name].update(att_dates)
    # Move processed files to archive
    file_source = zoom_path + '/' + img
    file_destination = zoom_path + '_archive/' + img
    os.replace(file_source, file_destination)

print(f'xlsx_data: {xlsx_data}')
xlsx_file_write(path, xlsx_data)


cv.waitKey(0)
