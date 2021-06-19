# Titanic

"Hello World" of Kaggle competition.

<!-- ################################################################################ -->

## How to run (with Docker)

1. Build docker image (if not build yet):  
   `./bin/build.sh`

2. Edit your python source `./share/*.py` and execution script `./share/entrypoint.sh`:  
   *  `./share/entrypoint.sh`,

3. Run docker container:  
   `./bin/run.sh`

4. [Optional] Clean-up docker image & container if you want:  
   * `./bin/clean.sh`      ... remove container only
   * `./bin/fullclean.sh`  ... remove image & container

<!-- ################################################################################ -->

## References

* Kabble "Titanic - Machine Learning from Disaster":  
  https://www.kaggle.com/c/titanic

* Docker Hub - python:  
  https://hub.docker.com/_/python
