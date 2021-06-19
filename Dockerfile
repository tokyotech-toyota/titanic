# base image
FROM python:3.7

# root work dir
WORKDIR /work

# install python packages via pip
COPY requirements.txt .
RUN pip install -r requirements.txt
