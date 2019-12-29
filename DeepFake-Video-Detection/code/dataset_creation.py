import cv2
import random
from affine_wrap import affine_wrap_face_swap

n = 1227

for i in range(1,n+1):
    scr_dir = "dataset/real/real_image ("+str(i)+").jpg"
    scr_img = cv2.imread(scr_dir)
    t = random.randint(1,n)
    dst_dir = "dataset/real/real_image ("+str(t)+").jpg"
    dst_img = cv2.imread(dst_dir)

    affine_wrap_face_swap(scr_img, dst_img, "dataset/fake/fake_image ("+str(i)+").jpg")
