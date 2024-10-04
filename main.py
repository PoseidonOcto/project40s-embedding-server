import tiktoken
from openai import OpenAI, RateLimitError
import time
from multiprocessing.dummy import Pool as ThreadPool
from pymilvus import MilvusClient
from flask import Flask, request
from markupsafe import escape
from flask_sqlalchemy import SQLAlchemy
from enum import Enum

app = Flask(__name__)

OPENAI_API_KEY = 'sk-svcacct-JHSMzMYNZRVwWuQk3kJ3d0K0F3EuJZP7XbCoI9lB6A6Q6zxbzL4F7PSjumV923F1uMqitGWgjuFV-sDsT3BlbkFJ2YGDJYbL73J-FXGLAZf_DgLACHHlJgH8OiWDTOnXPKyIjtCGUdPALJ3e4g5iIigyHoceAjm_yVgKGbcA'
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)

# 2. Set up the name of the collection to be created.
COLLECTION_NAME = 'claims'

# 3. Set up the dimension of the embeddings.
DIMENSION = 1536

# 5. Set up the connection parameters for your Zilliz Cloud cluster.
URI = 'https://in03-4bf6e70f6c36dab.serverless.gcp-us-west1.cloud.zilliz.com'
TOKEN = 'c8276ba3c7f4f1921f386b8d99fcf34f268fe89d00c320a5d679e4b943e09e01c6da86463a2ae370e07e208a297de1d585ae2aac'

MODEL_NAME = "text-embedding-3-small"
# Max tokens for text-embedding-3-small
# Source: https://zilliz.com/ai-models/text-embedding-3-small
MAX_TOKENS = 8191

SEARCH_BATCH_SIZE = 10  # Max batch size for zilliz api
DEFAULT_SEARCH_SIMILARITY_THRESHOLD = 0.6

RAW_CLAIM_DATA = 'deco3801-data.json'

# 6. Set up the connection parameters for your database.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

# Define the Fact table (Table 1)
class Fact(db.Model):
    __tablename__ = 'fact'
    # Composite key: 'id of user' and 'id of claim'
    user_id = db.Column(db.Integer, primary_key=True)
    claim_id = db.Column(db.Integer, primary_key=True)
    
    url = db.Column(db.Text, nullable=False)
    triggering_text = db.Column(db.Text, nullable=False)
    date_triggered = db.Column(db.Integer, nullable=False)  # milliseconds past epoch

# Define the Interaction table (Table 2)
class Interaction(db.Model):
    __tablename__ = 'interaction'
    # Composite key: 'id of user' and 'url'
    user_id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(255), primary_key=True)
    
    duration_spent = db.Column(db.Integer, nullable=False)  # duration in seconds
    date_spent = db.Column(db.Integer, nullable=False)  # milliseconds past epoch
    clicks = db.Column(db.Integer, nullable=False)

# Define the PoliticalLeaning enum
class PoliticalLeaningEnum(Enum):
    LEFT = 'left'
    CENTER_LEFT = 'center_left'
    CENTER = 'center'
    CENTER_RIGHT = 'center_right'
    RIGHT = 'right'

# Define the PoliticalLeaning table (Table 3)
class PoliticalLeaning(db.Model):
    __tablename__ = 'political_leaning'
    url = db.Column(db.String(255), primary_key=True)
    leaning = db.Column(db.Enum(PoliticalLeaningEnum), nullable=False)

# Intialise the database
with app.app_context():
    db.create_all()

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


@app.route("/")
def hello_world():
    return "<p>This is an api.</p>"


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

if __name__ == '__main__':
    # Will not run when launched as server.
    print(search(["drinking water on an empty stomach can make your face glow"] * 12,
                 DEFAULT_SEARCH_SIMILARITY_THRESHOLD))
