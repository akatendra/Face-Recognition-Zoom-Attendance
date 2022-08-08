# FaceRecognitionZoomAttendance #
Marking attendees at a ZOOM conference via face recognition.
***
### The project is built on libraries: ###
Face recognition

[https://github.com/ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)

Opencv-python

[https://github.com/opencv/opencv-python](https://github.com/opencv/opencv-python)
***
Folders:

1. class_got
>Folder with reference images of class members. The folder also contains files: 
>>1. With information about recognizing faces of class participants **face_encodings.npy**
>>2. File with the names of class participants (names are taken from the names of the reference images)
>>3. xlsx-file in which the presence of ZOOM conference is noted.
2. zoom_got
>In this folder, place a screenshot of the ZOOM conference where you want to mark the attendees
3. zoom_got_archive
>Folder into which the processed images are moved from zoom_got folder. On these images the presence is already marked
4. zoom_got_landmarks
>This folder contains the image for use in the program **attendance_cv2_thread2_landmarks.py**.
***

Files:
5. attendance_cv2_10_got.py
>The program processes one or more screenshots with different dates in their names. Information about the presence of the class participant is recorded in the file **attendance.xlsx**.

6. attendance_cv2_thread2_landmarks.py
>The program processes a single screenshot from the zoom_got_landmarks folder and superimposes control points on the image for face recognition. Three images are generated:
>>1. Image without control points
>>2. Control points are applied
>>3. Control points are connected by lines

4. classes.py
>The file contains the classes necessary for the operation of the programs

8. make_screenshot.py
>A program that takes a screenshot with the ZOOM conference. In the beginning it is necessary to choose in which folder to take a screenshot. After selecting the folder you have 5 seconds to select the window from which you want to take a screenshot. After 5 seconds the screenshot of the active window will be made.

9. setup_class.py
>A program that processes reference images of ZOOM class participants for subsequent recognition of those present at the ZOOM conference. The program generates three files:
>>- face_encodings.npy
>>>The result of reference pictures processing. Contains information about faces. 128 points per face.
>>- names.npy
>>>Names are taken from the names of the reference picture files for attendance
>>- attendance.xlsx
>>>File template for attendance marking
>These three files are placed in the **class_got** folder.

10. xlsx.py
>Contains the necessary functions for working with xlsx files. The openpyxl library is used
