# Named Entity Recognition (NER) API

API for performing named entity recognition from text input in Finnish. 
The model was trained by fine-tuning a [Finnish BERT language model](https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1)
to recognize 10 named entity categories:

- PERSON (person names)
- ORG (organizations)
- LOC (locations)
- GPE (geopolitical locations)
- PRODUCT (products)
- EVENT (events)
- DATE (dates)
- JON (Finnish journal numbers (diaarinumero))
- FIBC (Finnish business identity codes (y-tunnus))
- NORP (nationality, religious and political groups)
  
## Model training and testing 

The code used for training the model is available [here](https://github.com/DALAI-hanke/BERT_NER). 
More information on the training data, model parameters and test results is available at the [HuggingFace page](https://huggingface.co/Kansallisarkisto/finbert-ner)
hosting the model.

## Running the API

The API code has been built using the [FastAPI](https://fastapi.tiangolo.com/) library. It can be run either in a virtual environment,
or in a Docker container. Instructions for both options are given below. 

The API downloads latest versions of the model files from [HuggingFace](https://huggingface.co/Kansallisarkisto/finbert-ner)
when the code is run. By default, the files are saved to `~/.cache/huggingface/hub/`. 

This path can be modified by exporting the environment variable `TRANSFORMERS_CACHE`. 
For example in bash shell type `export TRANSFORMERS_CACHE=/path/to/cache` before running the code.

### Output format

The model makes predictions for named entities in the IOB2-format, where the B-prefix is used for the first token of 
an entity, and I-prefix for all subsequent tokens belonging to the same entity. 

By default, model output follows the input format, which is based on [wordpiece tokenization](https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt). Therefore, for example the input
sentence 'Helsingistä tuli Suomen suuriruhtinaskunnan pääkaupunki vuonna 1812.' produces the output

`[{'entity': 'B-GPE', 'score': 0.9999044, 'index': 1, 'word': 'Helsingistä', 'start': 0, 'end': 11}, {'entity': 'B-GPE', 'score': 0.9991748, 'index': 3, 'word': 'Suomen', 'start': 17, 'end': 23}, {'entity': 'I-GPE', 'score': 0.9968881, 'index': 4, 'word': 'suuri', 'start': 24, 'end': 29}, {'entity': 'I-GPE', 'score': 0.9972023, 'index': 5, 'word': '##ru', 'start': 29, 'end': 31}, {'entity': 'I-GPE', 'score': 0.99688524, 'index': 6, 'word': '##htina', 'start': 31, 'end': 36}, {'entity': 'I-GPE', 'score': 0.99559337, 'index': 7, 'word': '##sku', 'start': 36, 'end': 39}, {'entity': 'I-GPE', 'score': 0.99525815, 'index': 8, 'word': '##nna', 'start': 39, 'end': 42}, {'entity': 'I-GPE', 'score': 0.99037445, 'index': 9, 'word': '##n', 'start': 42, 'end': 43}, {'entity': 'B-DATE', 'score': 0.999951, 'index': 11, 'word': 'vuonna', 'start': 56, 'end': 62}, {'entity': 'I-DATE', 'score': 0.9998229, 'index': 12, 'word': '18', 'start': 63, 'end': 65}, {'entity': 'I-DATE', 'score': 0.9999138, 'index': 13, 'word': '##12', 'start': 65, 'end': 67}]`

This is a list of dictionaries, where each dictionary containsthe following keys and values:

- `entity`: Defines the predicted entity group of the token, using the IOB2 schema.
- `score`: Confidence score that the model gives to the prediction.
- `index`: Index of the token in the tokenized text input.
- `word`: Token / wordpiece for which the prediction is made. In the above example, for instance the word 'suuriruhtinaskunnan' is split into six wordpieces,
  where the 


### Running the API in a virtual environment

These instructions use a conda virtual environment, and as a precondition you should have Miniconda or Anaconda installed on your operating system. 
More information on the installation is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 

#### Create and activate conda environment using the following commands:

`conda create -n ner_api_env python=3.7`

`conda activate ner_api_env`

#### Install dependencies listed in the *requirements.txt* file:

`pip install -r requirements.txt`

#### Start the API running a single process (with Uvicorn server):

Using default host: 0.0.0.0, default port: 8000

`uvicorn api:app`
 
Select different host / port:

`uvicorn api:app --host 0.0.0.0 --port 8080`

#### You can also start the API with Gunicorn as the process manager (find more information [here](https://fastapi.tiangolo.com/deployment/server-workers/)):

`gunicorn api:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080`

  - workers: The number of worker processes to use, each will run a Uvicorn worker

  - worker-class: The Gunicorn-compatible worker class to use in the worker processes

  - bind: This tells Gunicorn the IP and the port to listen to, using a colon (:) to separate the IP and the port

### Running the API using Docker

As a precondition, you should have Docker Engine installed. More information on the installation can be found [here](https://docs.docker.com/engine/install/). 

#### Build Docker image using the *Dockerfile* included in the repository: 

`docker build -t ner_image .`

Here the new image is named ner_image. After successfully creating the image, you can find it in the list of images by typing `docker image ls`.

#### Create and run a container based on the image:

`sudo docker run -d --name ner_container -p 8000:8000 ner_image`

In the Dockerfile, port 8000 is exposed, meaning that the container listens to that port. In the above command, the corresponding host port can be chosen as the first element in `-p <host-port>:<container-port>`. If only the container port is specified, Docker will automatically select a free port as the host port. 
The port mapping of the container can be viewed with the command `sudo docker port postit_container`

## Testing the API

The API has one endpoint, `/ner`,  which expects the input text to be included in the client's POST request.

### Input format

The input text is expected to be in a json format, where the key 'text' is used for defining the content:

`'{"text": "Example text in Finnish."}'`


### Testing the API in a virtual environment

You can test the API for example using curl:

`curl -d '{"text": "Helsingistä tuli Suomen suuriruhtinaskunnan pääkaupunki vuonna 1812."}' -H "Content-Type: application/json" -X POST http://127.0.0.1:8000/ner`

The host and port should be the same ones that were defined when starting the API.

### Testing the API using Docker

In the Docker version of the API, it is easiest to use the `/postit` endpoint of the API. This can be tested 
for example using curl:

`curl http://127.0.0.1:8000/postit -F file=@/path/img.jpg`

### Output of the API

The output is in a .json form and consists of the predicted class label and the confidence for the prediction.
So for instance the output could be 

`{"prediction":"post-it","confidence":0.995205283164978}`


