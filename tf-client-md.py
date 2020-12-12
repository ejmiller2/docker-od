#! /usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image
import numpy as np
import requests
from pprint import pprint
import time
import os
import sys
import pdb
from collections import defaultdict

# image = PIL.Image.open("IMG_1441.JPG")
# image = PIL.Image.open("IMG_7556.jpg")
# image_np = np.array(image)

#-------------------------------------------------------------------------------
md_ids = [1, 2]
md_classes = ["animal", "person"]
classes_d = defaultdict ()
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def infer_image (filename, min_score=0.7) :
	have_display = "DISPLAY" in os.environ
	image = PIL.Image.open(filename)
	image_np = np.array(image)
	# print("Creating Payload")
	print("... ", end='')
	payload = {"instances": [image_np.tolist()]}
	#start = time.perf_counter()
	# print("Sending request...")
	print("...")
	res = requests.post("http://localhost:8080/v1/models/default:predict", json=payload)
	#print(f"Took {time.perf_counter()-start:.2f}s")
	# pprint(res.json())

	boxes = res.json()['predictions'][0]['detection_boxes']
	scores = res.json()['predictions'][0]['detection_scores']
	classes = res.json()['predictions'][0]['detection_classes']
	detections = int(res.json()['predictions'][0]['num_detections'])

	dim = image_np.shape
	width = dim[1]
	height = dim[0]
	if (have_display) :
		fig,ax = plt.subplots(1)
		# Display the image
		ax.imshow(image_np)

	for i in range(detections):
		label = classes_d [int (classes[i])]
		print('  Class', label, 'Score', scores[i])
		if (scores[i] < min_score):
			print('Scores too low')
			break
		ymin = int(boxes[i][0] * height)
		xmin = int(boxes[i][1] * width)
		ymax = int(boxes[i][2] * height)
		xmax = int(boxes[i][3] * width)
		b_width = xmax-xmin
		b_height = ymax-ymin
		print('\t', boxes[i], end='')
		print(xmin, ymin, b_width, b_height)
		# Create a Rectangle patch
		rect = patches.Rectangle((xmin,ymin),b_width,b_height,linewidth=1,edgecolor='r',facecolor='none')
		# Add the patch to the Axes
		if (have_display) :
			ax.add_patch(rect)

	if (have_display) :
		plt.show()

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def main (argv) :
	# pdb.set_trace ()
	for i in range (len (md_ids)) :
		classes_d[md_ids[i]] = md_classes[i]
	for arg in argv[1:] :
		print ('image: ', arg)
		infer_image (arg)

if __name__ == "__main__":
	main (sys.argv)
