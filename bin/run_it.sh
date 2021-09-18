#!/bin/bash

mountdir=$(dirname $(realpath $0))/../share
docker run --rm -it --name=$(whoami)-titanic --entrypoint="./share/entrypoint_it.sh" --mount type=bind,source=$mountdir,destination=/work/share $(whoami)/titanic
