import cv2, glob

for camera in glob.glob("/dev/video?"):
    c = cv2.VideoCapture(camera)
    if cap:
       print('Warning: unable to open video source: ', source)