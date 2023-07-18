from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseSettings
from transformers import pipeline 
import numpy as np
import uvicorn
import logging
import sys

# For logging options see
# https://docs.python.org/3/library/logging.html
logging.basicConfig(filename='api_log.log', filemode='w', format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

# Defines whether and how entity tags and corresponding tokens are aggreagated by the pipeline
# options: 'none', 'simple', 'first', 'average', 'max' and 'custom'
# For more information, see 
# https://huggingface.co/transformers/v4.7.0/_modules/transformers/pipelines/token_classification.html
class Settings(BaseSettings):
    aggregation_strategy: str = 'first'

settings = Settings()

try:
    # Initialize API Server
    app = FastAPI()
except Exception as e:
    logging.error('Failed to start the API server: %s' % e)
    sys.exit(1)

# Function is run (only) before the application starts
@app.on_event("startup")
async def load_model():
    """
    Load the pretrained model on startup.
    """
    # For aggregation strategy 'custom', model output is given in raw format
    aggregation = settings.aggregation_strategy if settings.aggregation_strategy != 'custom' else 'none'
    try:
        # Load tokenizer, model and the trained weights from HuggingFace Hub
        # By default, the files are saved to ~/.cache/huggingface/hub/
        model = pipeline(
                "token-classification", 
                model="Kansallisarkisto/finbert-ner", 
                aggregation_strategy=aggregation, 
                framework='pt', # PyTorch is used as the model framework
                device=-1 # By default CPU is used (-1), but also GPU id (>=0) can be given
            )
        # Add model to app state
        app.package = {"model": model}
    except Exception as e:
        logging.error('Failed to load the model files: %s' % e)
        raise HTTPException(status_code=500, detail='Failed to load the model files: %s' % e)


def transform_score(predictions_list):
    """Transforms the 'score' value to a format acceptable to FastAPI."""
    for item in predictions_list:
        item.update(score = np.float64(item['score']))

    return predictions_list


def filter_tags_scores(predictions_list):
    """Loops over the predictions and combines tokenized word pieces while preserving the I- and B-tags."""
    predictions = []
    n = len(predictions_list)
    token, tag, score, start, end, count = '', '', 0, 0, 0, 0
    # Loop over predictions
    for i, item in enumerate(predictions_list):
        # First token in the list
        if count == 0:
            token, tag, score, start, end, count = item['word'], item['entity'], item['score'], item['start'], item['end'], 1
            # The last prediction in the list is saved
            if i == n - 1:
                predictions.append({'entity_group': tag, 'score': score / count, 'word': token, 'start': start, 'end': end})
        else:
            # Checks if the i:th token is a continuation of the previous one
            if (item['entity'][2:] == tag[2:]) and (item['word'][:2] == '##'):
                token += item['word'][2:]
                score += item['score']
                end = item['end']
                count += 1
                # The last prediction in the list is saved
                if i == n - 1:
                    predictions.append({'entity_group': tag, 'score': score / count, 'word': token, 'start': start, 'end': end})
            else:
                # When token/tag changes, previous token, tag and score are saved
                predictions.append({'entity_group': tag, 'score': score / count, 'word': token, 'start': start, 'end': end})
                token, tag, score, start, end, count = item['word'], item['entity'], item['score'], item['start'], item['end'], 1
                # The last prediction in the list is saved
                if i == n - 1:
                    predictions.append({'entity_group': tag, 'score': score / count, 'word': token, 'start': start, 'end': end})

    return predictions


def predict(text):
    # Get model from app state
    model = app.package["model"]
    predictions_list = model(text)
    # If separate B- and I-tags are included in the output, aggregation is performed using a custom function
    if settings.aggregation_strategy == 'custom':
        predictions = filter_tags_scores(predictions_list)
    # For merged B- and I-tags, the pipeline is used for aggregation
    else:
        predictions = transform_score(predictions_list) 

    return predictions


# Endpoint for POST requests: input text is received with the http request
@app.post("/ner")
async def postit(text: str = Body(..., embed=True)):
    # Get predicted class and confidence
    try: 
        predictions = predict(text)
    except Exception as e:
        logging.error('Failed to analyze the input text: %s' % e)
        raise HTTPException(status_code=500, detail='Failed to analyze the input text: %s' % s)

    return predictions


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
