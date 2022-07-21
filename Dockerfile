FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
##NVIDIA_DRIVER_VERSION='nvidiaDriverVersion: "470.82.01"
##FROM python:3.9.0 
##python:3.10.1-buster
## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER author@example.com

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
