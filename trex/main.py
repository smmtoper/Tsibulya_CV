import cv2
import numpy as np
import pyautogui
import time
import mss

pyautogui.PAUSE = 0

slv = {'top': 280, 'left': 120, 'width': 740, 'height': 200}
right, left = 220, 120

sct = mss.mss()

while True:
    img = cv2.cvtColor(np.array(sct.grab(slv)), cv2.COLOR_BGRA2GRAY)
    img = cv2.resize(img, (500, 120))

    cropped_img = img[88, left:right]

    if 83 in cropped_img:
        pyautogui.keyUp('down')
        pyautogui.keyDown('up')
    else:
        pyautogui.keyUp('up')
        pyautogui.keyDown('down')

    time.sleep(0.1)
