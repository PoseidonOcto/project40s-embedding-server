from contextlib import contextmanager

import tiktoken
from openai import OpenAI, RateLimitError
import time
from multiprocessing.dummy import Pool as ThreadPool
from pymilvus import MilvusClient
from flask import Flask, request
from markupsafe import escape
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from enum import Enum
import os

from media_bias_insert import get_data_by_url

"""
API Server handling data storage and 'similar claim detection' for project40s.
Note that this file should not be made public while it holds private API keys.
Private API keys could instead be stored in environment variables on Railway if this repo must be made public.
"""
app = Flask(__name__)

# Zilliz Cloud cluster vector database
URI = 'https://in03-4bf6e70f6c36dab.serverless.gcp-us-west1.cloud.zilliz.com'
TOKEN = 'c8276ba3c7f4f1921f386b8d99fcf34f268fe89d00c320a5d679e4b943e09e01c6da86463a2ae370e07e208a297de1d585ae2aac'
# Set up the name of the collection to be created.
COLLECTION_NAME = 'claims'
# Set up the dimension of the embeddings.
DIMENSION = 1536

# OpenAI embedding API
OPENAI_API_KEY = 'sk-svcacct-JHSMzMYNZRVwWuQk3kJ3d0K0F3EuJZP7XbCoI9lB6A6Q6zxbzL4F7PSjumV923F1uMqitGWgjuFV-sDsT3BlbkFJ2YGDJYbL73J-FXGLAZf_DgLACHHlJgH8OiWDTOnXPKyIjtCGUdPALJ3e4g5iIigyHoceAjm_yVgKGbcA'
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "text-embedding-3-small"
# Max tokens for text-embedding-3-small
# Source: https://zilliz.com/ai-models/text-embedding-3-small
MAX_TOKENS = 8191

SEARCH_BATCH_SIZE = 10  # Max batch size for Zilliz api
DEFAULT_SEARCH_SIMILARITY_THRESHOLD = 0.6

# Postgres database hosted on railway, managed via Flask-SQLAlchemy.
# DATABASE_URL = "postgresql://postgres:ARwfipSWhFFMhyyuJRNXgbagWUjmyriE@junction.proxy.rlwy.net:58065/railway"
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Optional, to suppress warnings

# This API endpoint should not be accessible to public.
INSERT_MEDIA_BIAS_DATA_PASSWORD = os.environ['DATABASE_PRIVILEGED_OP_PASSWORD']
INSERT_MEDIA_BIAS_RAW_DATA = "data/media_bias.json"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Database creation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Base(DeclarativeBase):
    pass


DB = SQLAlchemy(model_class=Base)
DB.init_app(app)


class PoliticalLeaningEnum(Enum):
    """ The keys of this enum must exactly match their corresponding value. """
    EXTREME_LEFT = 'EXTREME_LEFT'
    LEFT = 'LEFT'
    LEFT_CENTER = 'LEFT_CENTER'
    CENTER = 'CENTER'
    RIGHT_CENTER = 'RIGHT_CENTER'
    RIGHT = 'RIGHT'
    EXTREME_RIGHT = 'EXTREME_RIGHT'
    CONSPIRACY = 'CONSPIRACY'
    PRO_SCIENCE = 'PRO_SCIENCE'
    SATIRE = 'SATIRE'


class Fact(DB.Model):
    """ Table storing 'fact-checks' the user has triggered when using the extension. """
    __tablename__ = 'fact'
    # Composite key: 'id of user' and 'id of claim'
    user_id = DB.Column(DB.Integer, primary_key=True)
    claim_id = DB.Column(DB.Integer, primary_key=True)

    url = DB.Column(DB.Text, nullable=False)
    triggering_text = DB.Column(DB.Text, nullable=False)
    date_triggered = DB.Column(DB.Integer, nullable=False)


class Interaction(DB.Model):
    """ Table storing user's media consumption. """
    __tablename__ = 'interaction'
    # Composite key: 'id of user' and 'url'
    user_id = DB.Column(DB.Integer, primary_key=True)
    url = DB.Column(DB.String(255), primary_key=True)

    duration_spent = DB.Column(DB.Integer, nullable=False)
    date_spent = DB.Column(DB.Integer, nullable=False)
    clicks = DB.Column(DB.Integer, nullable=False)


class PoliticalLeaning(DB.Model):
    """ Table storing political leanings of various websites, sourced from TODO """
    __tablename__ = 'political_leaning'

    # Primary key: 'url'
    url = DB.Column(DB.String(255), primary_key=True)

    leaning = DB.Column(DB.Enum(PoliticalLeaningEnum), nullable=False)


# Intialise the database
with app.app_context():
    DB.create_all()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Database - API Routes and helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class InvalidRequest(Exception):
    pass


def get_or_throw(request_data, key):
    if key in request_data:
        return request_data[key]
    raise InvalidRequest(f'Request is missing the field: "{key}"')


def get_or_throw_enum(request_data, key, enum):
    if key not in request_data:
        raise InvalidRequest(f'Request is missing the field: "{key}"')

    try:
        return enum[request_data[key]]
    except KeyError:
        raise InvalidRequest(
            f"The request field '{key}' has value '{request_data[key]}' which is not a valid member of {str(enum)}")


def handle_invalid_request(func):
    def wrapper(*args, **kwargs):
        try:
            data = func(*args, **kwargs)
        except InvalidRequest as e:
            return {
                'status': 'error',
                'message': str(e),
            }

        return {
            'status': 'success',
            'data': data
        }

    return wrapper


@handle_invalid_request
def check_password(request_data):
    if get_or_throw(request_data, 'password') != INSERT_MEDIA_BIAS_DATA_PASSWORD:
        raise InvalidRequest('The password provided was incorrect.')


@contextmanager
def rollback_on_err():
    try:
        yield
    except Exception:
        DB.session.rollback()
        raise
    else:
        DB.session.commit()


@app.route("/bias", methods=["POST"])
def insert_media_bias_data():
    request_data = request.get_json()

    response = check_password(request_data)
    if response['status'] == 'error':
        return response

    with open(INSERT_MEDIA_BIAS_RAW_DATA) as f:
        data = get_data_by_url(f)

        # Verify all bias ratings have been captured in schema
        valid_bias_ratings = set([x.value for x in PoliticalLeaningEnum])
        for bias_rating in data.values():
            assert bias_rating in valid_bias_ratings

        with rollback_on_err():
            # Delete all entries in table
            DB.session.query(PoliticalLeaning).delete()

            # Add new entries
            for url, bias_rating in data.items():
                DB.session.add(PoliticalLeaning(url=url, leaning=bias_rating))

    return {
        'status': 'success',
    }


@app.route("/recreate", methods=["POST"])
def recreate_tables():
    request_data = request.get_json()

    response = check_password(request_data)
    if response['status'] == 'error':
        return response

    DB.drop_all()
    DB.create_all()

    return {
        'status': 'success',
    }


@app.route("/users/create", methods=["POST"])
def user_create():
    request_data = request.get_json()
    try:
        entry = PoliticalLeaning(
            url=get_or_throw(request_data, 'url'),
            leaning=get_or_throw_enum(request_data, 'leaning', PoliticalLeaningEnum),
        )
        DB.session.add(entry)
        DB.session.commit()
        return {
            'status': 'success',
        }
    except InvalidRequest as e:
        return {
            'status': 'error',
            'message': str(e),
        }


@app.route("/get_all", methods=["POST"])
def get_all_data():
    try:
        result = DB.session.execute(DB.select(PoliticalLeaning).order_by(PoliticalLeaning.url)).all()
        formatted_result = [(row[0].url, row[0].leaning) for row in result]
        return {
            'status': 'success',
            'data': formatted_result,
        }
    except InvalidRequest as e:
        return {
            'status': 'error',
            'message': str(e),
        }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fact Checking - API Routes and helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def batch_claims(claims: list) -> list[list]:
    token_encoder = tiktoken.encoding_for_model(MODEL_NAME)

    i = 0
    tokens_in_batch = 0
    batches = []
    batch = []
    while i < len(claims):
        tokens_for_claim = len(token_encoder.encode(claims[i]))
        if tokens_in_batch + tokens_for_claim < MAX_TOKENS:
            batch.append(claims[i])
            tokens_in_batch += tokens_for_claim
            i += 1
        else:
            # If batch is empty, one claim is likely too big, and this will loop forever.
            assert batch != [] and tokens_for_claim < MAX_TOKENS
            print(tokens_in_batch)

            batches.append(batch)
            batch = []
            tokens_in_batch = 0

    # Add claims from last batch
    if batch:
        batches.append(batch)

    return batches


def openai_encoder(claims: list) -> list:
    while True:
        try:
            result = [entry.embedding
                      for entry in OPENAI_CLIENT.embeddings.create(input=claims, model=MODEL_NAME).data]
            break
        except RateLimitError:
            print("Hit rate limit: sleeping")
            time.sleep(100)

    return result


def embed_claims(claims: list) -> list:
    """
    Although parallel processing with more threads would significantly speeds up embedding,
    it would cause us to exceed our rate limit on the API. Additionally, catching the rate
    limit error and retrying led to losing connection with the openai sdk.

    I'm not sure why we kept losing connection, but it is an issue that arose when
    combining parallel processing, the openai sdk and 'time.sleep'.
    """
    batches = batch_claims(claims)

    # Make the Pool of workers
    pool = ThreadPool(4)  # Too many workers will flood gpt with too many requests

    # Encode on seperate threads and return the results
    results = pool.map(openai_encoder, batches)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Flatten results for each thread
    return [result for thread_results in results for result in thread_results]


def search_batch(client, embedded_claims: list, similarity_threshold: float):
    assert len(embedded_claims) <= 10

    return client.search(
        collection_name=COLLECTION_NAME,
        data=embedded_claims,
        limit=3,
        output_fields=['claim', 'author_name', 'author_url', 'review', 'url'],
        search_params={
            "metric_type": "COSINE",
            "params": {"radius": similarity_threshold}
        }
    )


def search(claims: list, similarity_threshold: float):
    """
    Response format loosely follows https://github.com/omniti-labs/jsend.
    """
    # Connect to Zilliz Cloud
    client = MilvusClient(
        # Public endpoint obtained from Zilliz Cloud
        uri=URI,
        token=TOKEN
    )

    # Get embedded claims
    embedded_claims = embed_claims(claims)

    # Batch the embedded claims for sending
    batches = [embedded_claims[i:i + SEARCH_BATCH_SIZE] for i in range(0, len(embedded_claims), SEARCH_BATCH_SIZE)]

    # Make the Pool of workers
    pool = ThreadPool(len(batches))

    # Encode on seperate threads and return the results
    results_by_thread = pool.map(lambda batch: search_batch(client, batch, similarity_threshold), batches)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Flatten results for each thread
    responses = [result for thread_results in results_by_thread for result in thread_results]

    assert len(claims) == len(responses)

    return {
        'status': 'success',
        'data': [{'claim': claim, 'responses': response} for (claim, response) in zip(claims, responses)],
    }


@app.route("/embedding-single/<query>")
def query_single(query):
    # Just to be safe. Source: https://flask.palletsprojects.com/en/3.0.x/quickstart/#html-escaping
    query = escape(query)
    if query.isspace():
        return {
            'status': 'error',
            'message': 'No query provided.',
        }

    # Note response will be a list with a single index.
    return search([query], DEFAULT_SEARCH_SIMILARITY_THRESHOLD)


@app.route("/embedding", methods=["POST"])
def query_multiple():
    request_data = request.get_json()
    if 'data' not in request_data:
        return {
            'status': 'error',
            'message': 'No claims provided',
        }
    if 'similarity_threshold' not in request_data:
        return {
            'status': 'error',
            'message': 'No similarity threshold provided',
        }

    return search(request_data['data'], request_data['similarity_threshold'])


@app.route("/")
def hello_world():
    return "<p>This is an api.</p>"


if __name__ == '__main__':
    # Will not run when launched as server.
    print(search(["drinking water on an empty stomach can make your face glow"] * 12,
                 DEFAULT_SEARCH_SIMILARITY_THRESHOLD))
