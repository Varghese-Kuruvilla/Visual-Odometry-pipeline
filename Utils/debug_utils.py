#A few utility function that will help you to debug
import cv2
import sys

def breakpoint():
    inp = input('Waiting for input...')

def display_image(winname,frame):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname,frame)
    key = cv2.waitKey(1)
    if (key & 0xFF == ord('q')):
        cv2.destroyAllWindows() 
        sys.exit()