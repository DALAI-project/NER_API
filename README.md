# Named Entity Recognition (NER) API

<img src="https://github.com/DALAI-hanke/NER_API/assets/33789802/315a9e94-3eb1-4b77-be66-5cec321a45f6.jpg"  width="60%" height="60%">


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

Different aggregation strategies can be used for changing the model output format. These can be changed with the parameter 
`AGGREGATION_STRATEGY` when starting the API. For example

`AGGREGATION_STRATEGY="simple" uvicorn api:app`

#### Aggregation strategy: 'none'

By default, model output follows the input format, which is based on [wordpiece tokenization](https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt). Therefore, for example the input
sentence 'Helsingistä tuli Suomen suuriruhtinaskunnan pääkaupunki vuonna 1812.', when the aggregation strategy 'none is used, produces the output

`[{'entity': 'B-GPE', 'score': 0.9999044, 'index': 1, 'word': 'Helsingistä', 'start': 0, 'end': 11}, {'entity': 'B-GPE', 'score': 0.9991748, 'index': 3, 'word': 'Suomen', 'start': 17, 'end': 23}, {'entity': 'I-GPE', 'score': 0.9968881, 'index': 4, 'word': 'suuri', 'start': 24, 'end': 29}, {'entity': 'I-GPE', 'score': 0.9972023, 'index': 5, 'word': '##ru', 'start': 29, 'end': 31}, {'entity': 'I-GPE', 'score': 0.99688524, 'index': 6, 'word': '##htina', 'start': 31, 'end': 36}, {'entity': 'I-GPE', 'score': 0.99559337, 'index': 7, 'word': '##sku', 'start': 36, 'end': 39}, {'entity': 'I-GPE', 'score': 0.99525815, 'index': 8, 'word': '##nna', 'start': 39, 'end': 42}, {'entity': 'I-GPE', 'score': 0.99037445, 'index': 9, 'word': '##n', 'start': 42, 'end': 43}, {'entity': 'B-DATE', 'score': 0.999951, 'index': 11, 'word': 'vuonna', 'start': 56, 'end': 62}, {'entity': 'I-DATE', 'score': 0.9998229, 'index': 12, 'word': '18', 'start': 63, 'end': 65}, {'entity': 'I-DATE', 'score': 0.9999138, 'index': 13, 'word': '##12', 'start': 65, 'end': 67}]`

This is a list of dictionaries, where each dictionary containsthe following keys and values:

- `entity`: Defines the predicted entity group of the token, using the IOB2 schema.
- `score`: Confidence score that the model gives to the prediction.
- `index`: Index of the token in the tokenized text input.
- `word`: Token / wordpiece for which the prediction is made. In the above example, for instance the word 'suuriruhtinaskunnan' is split into six wordpieces,
  where the pieces following the first one begin with '##'.
- `start`: Index of the start of the token/wordpiece.
- `end`: Index of the end of the token/wordpiece.

#### Aggregation strategy: 'simple'

This aggregation strategy groups together the B- and I-parts of the same entities into a single entity. Now the output for the example sentence becomes:

`[{'entity_group': 'GPE', 'score': 0.9999044, 'word': 'Helsingistä', 'start': 0, 'end': 11}, {'entity_group': 'GPE', 'score': 0.995911, 'word': 'Suomen suuriruhtinaskunnan', 'start': 17, 'end': 43}, {'entity_group': 'DATE', 'score': 0.9998959, 'word': 'vuonna 1812', 'start': 56, 'end': 67}]`

Now for example the word 'suuriruhtinaskunnan' is one token belonging to entity group 'GPE'. Token/wordpiece index is omitted from the results. More information on the 'simple' strategy and its variations ('first', 'average', 'max') can be found [here](https://huggingface.co/transformers/v4.7.0/_modules/transformers/pipelines/token_classification.html). By default, the 'first' strategy is used in the API. 

#### Aggregation strategy: 'custom'

This aggregation option is custom built, and is not part of the transformers-library. The goal is to group together wordpieces 
belonging to a single B- or I-tag, so that the aggregation preserves the IOB2-style annotation format. The output for the example sentence is:
 
`[{"entity_group":"B-GPE","score":0.9999043941497803,"word":"Helsingistä","start":0,"end":11},{"entity_group":"B-GPE","score":0.9991747736930847,"word":"Suomen","start":17,"end":23},{"entity_group":"I-GPE","score":0.9953669706980387,"word":"suuriruhtinaskunnan","start":24,"end":43},{"entity_group":"B-DATE","score":0.9999510049819946,"word":"vuonna","start":56,"end":62},{"entity_group":"I-DATE","score":0.9998683929443359,"word":"1812","start":63,"end":67}]`

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

`sudo docker build -t ner_image .`

Here the new image is named ner_image. After successfully creating the image, you can find it in the list of images by typing `docker image ls`.

#### Create and run a container based on the image:

`sudo docker run -d --name ner_container -p 8000:8000 ner_image`

In the Dockerfile, port 8000 is exposed, meaning that the container listens to that port. In the above command, the corresponding host port can be chosen as the first element in `-p <host-port>:<container-port>`. If only the container port is specified, Docker will automatically select a free port as the host port. 
The port mapping of the container can be viewed with the command `sudo docker port postit_container`

If you want to change the default aggregation strategy ('simple') when creating the container, this can be done by using the -e flag:

`sudo docker run -d --name ner_container -p 8000:8000 -e AGGREGATION_STRATEGY="custom" ner_image`

## Logging

Logging events are saved into a file `api_log.log` in the same folder where the `api.py` file is located. Previous content of the log file is overwritten after each restart. More information on different logging options is available [here](https://docs.python.org/3/library/logging.html).

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

The Docker version of the API can bes tested (when the container is running) for example with curl using the same arguments as above. 



