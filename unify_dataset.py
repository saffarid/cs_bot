import os
import cv2 as cv

dataset_source = "./datasets/cs_go/source/src"
dataset_unify = "./datasets/cs_go/source/unify"
cs_go_src_name = "{0}/cs_go_unify_{1}.jpg"

new_shape = (1280, 700)

clear = lambda: os.system('cls')

def main():
    print("Unification of the image from dataset has begun")

    imgs = os.listdir(dataset_source)

    if not os.path.exists(dataset_unify):
        os.mkdir(dataset_unify)

    for indx, s_img in enumerate(imgs):
        img = cv.imread(f"{dataset_source}/{s_img}")
        img = cv.resize(img, new_shape)
        cv.imwrite(cs_go_src_name.format(dataset_unify, indx+1), img)
        if indx != len(imgs) - 1:
            end = '\r'
        else:
            end = '\n'
        print(f"\r\b{indx+1}/{len(imgs)}", end=end)

    print("Images from dataset are unified")

if __name__ == "__main__":
    main()
