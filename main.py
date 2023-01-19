import random
import mss
import cv2 as cv
import numpy as np
import time

from ultralytics import YOLO as yolo

sct = mss.mss()

right_monitor = sct.monitors[1]
left_monitor = sct.monitors[2]
l_mon = (0, 0, left_monitor["width"], left_monitor["height"])
model = yolo("yolov8n.yaml")  # загрузите предварительно обученную модель YOLOv8n


def grab_screen():
    img = np.asarray(sct.grab(l_mon))
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return img


def gen_border_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def start_end_from_tensor(x):
    return (int(x[0].item()), int(x[1].item())), (int(x[2].item()), int(x[3].item()))


def screen_record():
    new_frame_time, prev_frame_time, fps = 0, 0, 0
    while True:

        img = grab_screen()
        img = cv.resize(img, (800, 600))
        results = model.predict(source=img)

        for result in results:
            xyxy = result.boxes.xyxy
            for xy in xyxy:
                start_point, end_point = start_end_from_tensor(xy)
                cv.rectangle(img, start_point, end_point, gen_border_color())

        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        cv.putText(img, str(fps), (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv.imshow('scr', img)
        if cv.waitKey(1) and 0xFF == ord("q"):
            cv.destroyAllWindows()


def main():
    model.train(data="coco128.yaml", epochs=100)
    screen_record()


if __name__ == "__main__":
    main()
