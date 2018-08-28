#!/bin/bash

#Check OS
if [[ $(uname) == 'Darwin' ]];
then
	DOCKERFILE="Dockerfile_keras_cpu"
	DOCKER="docker"
	BASE_FOLDER="${HOME}/Documents/Practicas/TFG/Repositorio/TFG"
	DATA_FOLDER="${HOME}/Documents/Datasets"
	DOC_FOLDER="${HOME}/Documents//U/TFG"
else
	DOCKERFILE="Dockerfile_keras_gpu"
	DOCKER="nvidia-docker"
	BASE_FOLDER="${HOME}/Documentos/TFG/Repositorio/TFG"
	DATA_FOLDER="${HOME}/Documentos/Datasets"
fi

#Esto crea una imagen con nombre unimodal_image
cat $DOCKERFILE | $DOCKER build -t keras_unimodal_image -

sudo $DOCKER run --name keras_container \
 -v $BASE_FOLDER:/TFG \
 -v $DATA_FOLDER:/Datasets \
 -v $DOC_FOLDER:/Docs \
 -p 8888:8888 \
 -p 6006:6006 \
 -it keras_unimodal_image bash
