import tiktoken
from openai import OpenAI, RateLimitError
import time
from multiprocessing.dummy import Pool as ThreadPool
from pymilvus import MilvusClient
from flask import Flask, request
from markupsafe import escape

app = Flask(__name__)

OPENAI_API_KEY = 'sk-svcacct-JHSMzMYNZRVwWuQk3kJ3d0K0F3EuJZP7XbCoI9lB6A6Q6zxbzL4F7PSjumV923F1uMqitGWgjuFV-sDsT3BlbkFJ2YGDJYbL73J-FXGLAZf_DgLACHHlJgH8OiWDTOnXPKyIjtCGUdPALJ3e4g5iIigyHoceAjm_yVgKGbcA'  # Use your own Open AI API Key here
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)

# 2. Set up the name of the collection to be created.
COLLECTION_NAME = 'claims'

# 3. Set up the dimension of the embeddings.
DIMENSION = 1536

# 4. Set up the number of records to process.
COUNT = 100

# 5. Set up the connection parameters for your Zilliz Cloud cluster.
URI = 'https://in03-4bf6e70f6c36dab.serverless.gcp-us-west1.cloud.zilliz.com'
TOKEN = 'c8276ba3c7f4f1921f386b8d99fcf34f268fe89d00c320a5d679e4b943e09e01c6da86463a2ae370e07e208a297de1d585ae2aac'

# Max length of claim
# MAX_LEN = 200

MODEL_NAME = "text-embedding-3-small"
# Max tokens for text-embedding-3-small
# Source: https://zilliz.com/ai-models/text-embedding-3-small
MAX_TOKENS = 8191

SLEEP_INCREMENT = 0.1

RAW_CLAIM_DATA = 'deco3801-data.json'


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


"""
Although parallel processing with more threads would significantly speeds up embedding, 
it would cause us to exceed our rate limit on the API. Additionally, catching the rate
limit error and retrying led to losing connection with the openai sdk.
 
I'm not sure why we kept losing connection, but it is an issue that arose when
combining parallel processing, the openai sdk and 'time.sleep'.
"""


def embed_claims(claims: list) -> list:
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
    return "<p>Hello, World!</p>"


"""
Response format loosely follows https://github.com/omniti-labs/jsend.
"""


def search(claims: list):
    # Connect to Zilliz Cloud
    client = MilvusClient(
        # Public endpoint obtained from Zilliz Cloud
        uri=URI,
        token=TOKEN
    )

    # Claim
    # query = "Incest is legal in New Jersey and Rhode Island."
    # query = "Shaquille O'Neal throws Tim Walz out of his Restaurant"
    # query = "drinking water on an empty stomach can make your face glow"

    search_params = {
        "metric_type": "COSINE",
        "params": {"radius": 0.6}
    }

    responses = client.search(
        collection_name=COLLECTION_NAME,
        data=embed_claims(claims),
        limit=3,
        output_fields=['claim', 'author_name', 'author_url', 'review', 'url'],
        search_params=search_params
    )

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
    return search([query])


@app.route("/embedding", methods=["POST"])
def query_multiple():
    request_data = request.get_json()
    if 'data' not in request_data:
        return {
            'status': 'error',
            'message': 'No claims provided',
        }

    return search(request_data['data'])



    # -----------------

    # Search for similar titles
    # def search(text):
    #     res = collection.search(
    #         data=[embed(text)],
    #         anns_field='embedding',
    #         param={"metric_type": "L2", "params": {"nprobe": 10}},
    #         output_fields=['title'],
    #         limit=5,
    #     )
    #
    #     ret = []
    #
    #     for hits in res:
    #         for hit in hits:
    #             row = []
    #             row.extend([hit.id, hit.distance, hit.entity.get('title')])
    #             ret.append(row)
    #
    #     return ret
    #
    #
    # search_terms = [
    #     'self-improvement',
    #     'landscape',
    # ]
    #
    # for x in search_terms:
    #     print('Search term: ', x)
    #     for x in search(x):
    #         print(x)
    #     print()


if __name__ == '__main__':
    search(["drinking water on an empty stomach can make your face glow"])
    # CLIENT.drop_collection(COLLECTION_NAME)
