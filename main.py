import random
import mss
import cv2 as cv
import numpy as np
import time

import torch


from ultralytics import YOLO as yolo


title_fps = "FPS"

sct = mss.mss() #Объект для захвата скриншотов

right_monitor = sct.monitors[1]
left_monitor = sct.monitors[2]
l_mon = (0, 0, left_monitor["width"], left_monitor["height"])
r_mon = (0, 0, right_monitor["width"], right_monitor["height"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = yolo("yolov8n.pt")  # загрузите предварительно обученную модель YOLOv8n


def grab_screen():
    img = np.asarray(sct.grab(l_mon))
    img = cv.cvtColor(cv.cvtColor(img, cv.COLOR_RGB2BGR), cv.COLOR_BGR2RGB)
    return img


def gen_border_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def start_end_from_tensor(x):
    return (int(x[0].item()), int(x[1].item())), (int(x[2].item()), int(x[3].item()))


def screen_record():
    new_frame_time, prev_frame_time, fps = 0, 0, 0
    while True:

        prev_frame_time = new_frame_time
        img = grab_screen()
        img = cv.resize(img, (800, 600))
        # img = np.moveaxis(img, -1, 0)

        results = model.predict(source=img, half=False, stream=True)

        new_frame_time = time.time()

        fps = int(1 / (new_frame_time - prev_frame_time))
        print(f"fps: {fps}; computed time: {new_frame_time - prev_frame_time}")


        for result in results:
            try:
                if result.boxes == None: continue
                xyxy = result.boxes.xyxy
                for xy in xyxy:
                    start_point, end_point = start_end_from_tensor(xy)
                    cv.rectangle(img, start_point, end_point, gen_border_color(), thickness=2)
            except AttributeError as er:
                print(er)

        cv.putText(img, str(fps), (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv.imshow(title_fps, img)
        if cv.waitKey(1) and 0xFF == ord("q"):
            cv.destroyAllWindows()


def main():
    # model.train(data="Argoverse.yaml", epochs=5)
    # model.export(format="onnx")

    print(device)
    screen_record()


if __name__ == "__main__":
    main()
