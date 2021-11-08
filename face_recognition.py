import cv2
import face_recognition
import os
import numpy as np
import sys

subjects = []
status = []
face_encoding_list = []


def colour_face(colorCode):
    if colorCode == "vip":
        return (0, 255, 0)
    if colorCode == "redlisted":
        return (0, 0, 255)
    else:
        return (255, 255, 255)


def draw_rectangle(img, rect, color):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (w, h), color, 2)


def draw_label(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def preprocessing(data_folder_path):
    global subjects
    global status

    dirs = os.listdir(data_folder_path)

    faces = []

    labels = []

    for dir_name in dirs:

        if not dir_name.startswith("person"):
            continue;

        label = int(dir_name.replace("person", ""))

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:

            if image_name.startswith("."):
                continue;

            if image_name == "name.txt":
                name_path = subject_dir_path + "/" + image_name
                with open(name_path, 'r+') as name:
                    content = name.read()
                    content = content.lower()
                    subjects.append(content)

            elif image_name == "status.txt":
                name_path = subject_dir_path + "/" + image_name
                with open(name_path, 'r+') as name:
                    content = name.read()
                    content = content.lower()
                    status.append(content)

            else:
                image_path = subject_dir_path + "/" + image_name
                image = face_recognition.load_image_file(image_path)

                print("Number of faces scanned: ", len(faces) + 1)
                cv2.waitKey(100)

                faces.append(image)
                labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces


faces = preprocessing("train_images")


def show_results(window):
    global faces
    global subjects
    global status
    global face_encoding_list

    face_locations = face_recognition.face_locations(window, number_of_times_to_upsample=2)
    try:
        for face in faces:
            face_encoding = face_recognition.face_encodings(face, num_jitters=2)[0]
            face_encoding_list.append(face_encoding)
    except IndexError:
        print("Uh oh.No face. Check the image files. Aborting...")
        quit()
    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    img = window.copy()

    for face_location in face_locations:
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                    right))

        face_image = frame[top:bottom, left:right]
        temp_face_encoding = face_recognition.face_encodings(face_image, num_jitters=2)
        if len(temp_face_encoding) > 0:
            face_encoding = temp_face_encoding[0]
            results = face_recognition.compare_faces(face_encoding_list, face_encoding, tolerance=0.6)
            if True in results:
                temp_index = results.index(True)
                colorCode = colour_face(status[temp_index])
                draw_rectangle(img, (left, top, right, bottom), colorCode)
                draw_label(img, subjects[temp_index], left, top - 5)
                print(subjects[temp_index])
                print(status[temp_index])

            else:
                print("not found")
                draw_rectangle(img, (left, top, right, bottom), (255, 255, 255))
                draw_label(img, "No Match", left, top - 5)

    cv2.imshow("Face detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


vid = cv2.VideoCapture(0)
while True:
    check, frame = vid.read()
    cv2.imshow("Press c to Capture frame , q to exit", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        show_results(frame)

vid.release()
cv2.destroyAllWindows()
