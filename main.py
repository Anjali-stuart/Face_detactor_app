import cv2
from random import randrange
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#image for face detactor
# img = cv2.imread('stefan_Elena_Demon.jpg')

# To capture video from webcam.
webcam = cv2.VideoCapture('agar tum sath ho.mp4')

# Iterate forever over frames
while True:
    ### Read the current frame
    successful_frame_read, frame = webcam.read()

    # must convert to grayscale, we can change the image to any color just learn it from cv documentation
    grayScaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayScaled_img)  # it will detect nose eyes as a ectangle
    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),
                      8)  # --> here  1) top left . 2) right bottom  3) color green 4) thickness

    cv2.imshow('anjali face detactor', frame)

    key = cv2.waitKey(1)

    ### stop if Q key is pressed
    if key == 81 or key == 113 :
        break


"""
this is the half of code for image detacting.
# print(face_coordinates)


# for i in range(len(face_coordinates)):
    # (x, y, w, h) = face_coordinates[i]
# cv2.imshow('anjali face detactor', img)
cv2.waitKey()
# hard cascade algo implementation so need to make image black and white.
"""
print("Code completed")
