import os
import numpy as np
from datetime import datetime
import cv2 as cv
import time

# Class WindowCapture from https://www.youtube.com/watch?v=WymCpVUPWQ4
import win32con
import win32gui
import win32ui
from win32api import GetSystemMetrics



def screen_capture():
    # screen properties
    width = GetSystemMetrics(0)
    height = GetSystemMetrics(1)
    hwnd = None

    # find the handle for the window we want to capture
    # hwnd = win32gui.FindWindow(None, window_name)

    # get the window image data
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (width, height), dcObj, (0, 0), win32con.SRCCOPY)

    # convert the raw data into a format opencv can read
    # dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    # free resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    # drop the alpha channel, or cv.matchTemplate() will throw an error like:
    #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type()
    #   && _img.dims() <= 2 in function 'cv::matchTemplate'
    img = img[..., :3]

    # make image C_CONTIGUOUS to avoid errors that look like:
    #   File ... in draw_rectangles
    #   TypeError: an integer is required (got type tuple)
    # see the discussion here:
    # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
    img = np.ascontiguousarray(img)

    return img

def save_screenshot():
    list_items = os.listdir()
    dirs = []
    # Finding only dir which name started from 'zoom'
    for item in list_items:
        if os.path.isdir(item):
            if item.find('zoom') != -1 and item.find('archive') == -1:
                dirs.append(item)
    # print(dirs)
    # Output list of directories to choose needed class directory
    count = 1
    for class_dir in dirs:
        print(f'{class_dir} - {count}')
        count += 1
    # Choose class directory to save screenshot
    choose_class_dir = int(input('Choose class (input class number): '))
    class_dir = dirs[choose_class_dir - 1]
    print(f'Directory choosed: {class_dir}')

    # Change the current directory
    # to specified directory
    os.chdir(class_dir)
    time.sleep(5)
    screenshot = screen_capture()

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'zoom_{dt_string}.jpg'

    # Save screenshot
    cv.imwrite(filename, screenshot)
    print(f'Screenshot saved to: {class_dir}/{filename}')

if __name__ == '__main__':
    # setup_class('class_beetroot')
    # setup_class('class_got')
    # screenshot = screen_capture()

    save_screenshot()
