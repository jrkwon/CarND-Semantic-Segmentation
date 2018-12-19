import cv2
import os
import sys

if len(sys.argv) == 1:
    print('Need runs_id')
    sys.exit()

runs_id = sys.argv[1] #'1545192834.8808994'
image_folder = 'runs/' + runs_id
video_name = 'video-'+runs_id+'.avi'

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()
