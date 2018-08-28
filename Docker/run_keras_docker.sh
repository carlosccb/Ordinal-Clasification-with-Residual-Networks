#!/bin/bash

#Check OS
if [[ $(uname) == 'Darwin' ]];
then
	DOCKER="docker"
else
	DOCKER="nvidia-docker"
fi

#Esto ejecuta el contenedor
#Hay que meter los datos en una carpeta que esta en /data/lisatmp4/beckhamc/
# Esta carpeta creo que tiene varias carpetas:
#  * adience_data/aligned_256x256 contiene las imagenes ?
#  * /hdf5/ contiene adience_256.h5 y adience_test_256.h5

$DOCKER start keras_container
$DOCKER exec -w /TFG -it keras_container /bin/bash