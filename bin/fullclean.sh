#!/bin/bash

docker container rm $(whoami)-titanic
docker image rm $(whoami)/titanic
