# base image
FROM python:3.7

# root work dir
WORKDIR /work

# install packages for apple M1 chip
RUN apt update -y && apt upgrade -y
RUN apt install -y libhdf5-dev

# install python packages via pip
COPY requirements.txt .
RUN pip install -r requirements.txt
