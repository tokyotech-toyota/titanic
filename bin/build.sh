#!/bin/bash
set -e

dockerfiledir=$(dirname $0)/../
docker image build --tag $(whoami)/titanic $dockerfiledir
