This project provides and example for running object detection models
in a Docker container using TensorFlow Serving.
By default, the assumption is no GPU acceleration. Some notes to work
with Nvidia GPUs are also included.

This README covers the following:
  * Build a Docker container using
    > a model from TensorFlow Model Zoo
    > Microsoft's MegaDetector
  * Run the Docker container
  * Run an example python client on a local image (assumes COCO model)
  * Managing Docker containers

------------------------------------------------------------
Building Docker containers
------------------------------------------------------------
May need to run docker commands with sudo

*** For models from TensorFlow Model Zoo ***

Get MODEL_URL at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
Command line to build (<MODEL_URL> from above, <TAG_NAME> of your choosing):
  docker build -t <TAG_NAME> --build-arg model_url=<MODEL_URL> .
Example:
  docker build -t tf-frcnn --build-arg model_url=http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz .
  
To run this with GPU support, use Dockerfile.gpu. Make sure you have 
Nvidia support enabled in Docker. This example uses nvidia-docker2
(see [instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)).
It was tested with CUDA 9.0 and tensorflow/serving:1.12.0-gpu. For 
later CUDA support, try a newer version of tensorflow/serving, or 
use the latest (tensorflow/serving:latest-gpu).
Example:
  docker build -t tf-frcnn-gpu --build-arg model_url=http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz .

*** For Microsoft MegaDetector ***

Get MODEL_URL for TFServing at https://github.com/microsoft/CameraTraps/blob/master/megadetector.md
Command line to build (<MODEL_URL> from above, <TAG_NAME> of your choosing):
  docker build -t <TAG_NAME> --build-arg model_url=<MODEL_URL> -f Dockerfile.mega .
Example:
  docker build -t tf-md --build-arg model_url=https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/saved_model_megadetector_v3_tf19.zip -f Dockerfile.mega .

------------------------------------------------------------
To run a Docker container:
------------------------------------------------------------
  docker run -p 8080:8080 -p 8081:8081 <TAG_NAME>
Examples:
  docker run -p 8080:8080 -p 8081:8081 tf-frcnn
  docker run -p 8080:8080 -p 8081:8081 tf-md

For GPU:
  docker run --gpus all -p 8080:8080 -p 8081:8081 <TAG_NAME>
Examples:
  docker run -p 8080:8080 -p 8081:8081 tf-frcnn-gpu
  
------------------------------------------------------------
Example Python Client
------------------------------------------------------------
Client for docker-od for COCO or MegaDetector models
  expects server to be running (see above)
Tested with python version 3.5
Images are specified as argument
If has display, script will show image with bounding box
  for each detected object
To run client (COCO):
  python tf-client-coco.py <images>
example:
  python tf-client-coco.py IMG_1441.JPG IMG_7556.jpg

To run client (MegaDetector - animal or person):
  python tf-client-md.py <images>
example:
  python tf-client-md.py IMG_1441.JPG IMG_7556.jpg

------------------------------------------------------------
To manage Docker containers
------------------------------------------------------------
List Docker images
  docker images -a
Remove Docker images
  docker rmi <TAG_NAME>
List Docker processes
  docker ps -a
Stop container
  docker stop <CONTAINER_ID>  # from docker ps -a
  docker stop 134bb13a96c9

Remove all exited Containers
  docker rm $(docker ps -a -f status=exited -q)
  docker rm 134bb13a96c9

Show states of running Containers
  docker stats
To see which tf image is running:
  sudo docker ps -a | awk '/tf-*/ {print $2}'
