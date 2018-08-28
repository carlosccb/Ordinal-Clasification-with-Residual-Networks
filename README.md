**READMEs**:
* English Readme :  [Undergraduate final project: Ordinal Clasification with Residual Networks for the Adience dataset](#undergraduate-final-project-ordinal-clasification-with-residual-networks-for-the-adience-dataset).
* Spanish Readme: [Trabajo Fin de Grado: Clasificación Ordinal mediante Redes Neuronales Residuales para el *dataset* Adience](#trabajo-fin-de-grado-clasificación-ordinal-mediante-redes-neuronales-residuales-para-el-dataset-adience)
_______________________________________________________

# [Undergraduate final project: Ordinal Clasification with Residual Networks for the Adience dataset](#undergraduate-final-project-ordinal-clasification-with-residual-networks-for-the-adience-dataset)

This repository contains the code created for my Undergraduate final project which consists on replicating the results of a paper, [Unimodal Probability Distributions for Deep Ordinal Classification](https://arxiv.org/pdf/1705.05278.pdf).
As the name sugests unimodal probability distributions are used to enforce ordinal clasifications on residual networks. To train and test the models created
the [Adience](https://www.openu.ac.il/home/hassner/Adience/data.html) dataset is used.

The resnet is created with Keras using the tensorflow backend.

## Experimentation
All the experiments were run on docker containers using nvidia-docker.
The scripts needed to create the docker container are in the `Docker` folder. The dockerfiles needed to create containers for either GPU or CPU are included.
The instalation of all the libraries related with NVIDIA's CUDA & cudNN must be done previously to the creation of the docker containers.

### Docker image instalation
1. Install NVIDIA's CUDA & cudNN if the code is gonna be run on a GPU. Otherwise don't install anything.
2. Create the container using the `Dockerfile_keras_*` accordingly to the previous step. My personal configuration for the container is on the *install_keras_docker.sh* script.
3. Lastly run the container created in the previous step and enjoy!

### Run the experimentation
The experiments with the described architectures are coded in the main file.
If the container differs from mine change the location of the dataset variables accordingly to the paths in your container. 
To run the experiments simply launch this file as any other python file. `python main.py`
The models will be trained and then tested. The best models according to the validation loss will be stored in the folder *./models*.
All the training information, csv of losses and accuracy in train and validation, and tensorboard summaries will loged int the *./logs* folder.

_______________________________________________________

# [Trabajo Fin de Grado: Clasificación Ordinal mediante Redes Neuronales Residuales para el *dataset* Adience](#trabajo-fin-de-grado-clasificación-ordinal-mediante-redes-neuronales-residuales-para-el-dataset-adience).


Este repositorio contiene el código creado para mi Trabajo Fin de Grado que consiste en replicar los resultados del artículo [Unimodal Probability Distributions for Deep Ordinal Classification](https://arxiv.org/pdf/1705.05278.pdf). Se utilizan distribuciones de probabilidad unimodales para imponer que la clasificación realizada por la red residual sea ordinal. Para entrenar y evaluar los modelos creados se usa el conjunto de datos [Adience](https://www.openu.ac.il/home/hassner/Adience/data.html).

La red residual ha sido creada con la librería Keras utilizando el backend de tensorflow.

## Experimentación
Todos los experimentos han sido ejecutados en contenedores de docker usando nvidia-docker.

Los script necesarios para crear el contenedor se encuentran en la carpeta `Docker`. Tanto los ficheros dockerfile para crear los contenedores para GPU o CPU están incluidos.
Se debe realizar la instalación de las librerías de NVIDIA *CUDA* y *cudNN* de forma previa a la creación de los contenedores.

### Instalación de la imagen de Docker
1. Si los experimentos se van a realizar sobre GPU instalar *CUDA* y *cudNN*, si no no hace falta.
2. Crear el contenedor utilizando el archivo `Dockerfile_keras_*` correspondiente al modo de ejecución elegido. Mi configuración personal para el contendor es el script *install_keras_docker.sh*.
3. Por último sólo queda ejecutar el contenedor creado.

### Ejecutar los experimentos
Los experimentos con las arquitecturas de red descritas en el artículo se encuentran en el fichero `main.py`.
Si el contenedor creado difiere del mío se tendrán que cambiar de manera acorde las variables que contienen el directorio del dataset.
Para lanzar los experimentos simplemente hay que ejecutar el fichero main: `python main.py`
Los modelos se entrenarán y posteriormente se evaluarán. Se guarda el modelo completo que sea mejor según la métrica de validación en la carpeta *./models*..

Toda la información relacionada con el entrenamiento se guardará en csv con las métricas de entrenamiento y validación y en ficheros de tensorboard en la carpeta *./logs*.
