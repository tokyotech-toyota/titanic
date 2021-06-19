#!/bin/bash

mountdir=$(dirname $(realpath $0))/../share
docker run --rm --name=$(whoami)-titanic --entrypoint="./share/entrypoint.sh" --mount type=bind,source=$mountdir,destination=/work/share $(whoami)/titanic
