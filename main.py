from contextlib import contextmanager
import tiktoken
from openai import OpenAI, RateLimitError
import time
from multiprocessing.dummy import Pool as ThreadPool
from pymilvus import MilvusClient
from flask import Flask, request
from markupsafe import escape
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import DeclarativeBase, aliased
from sqlalchemy import and_, func
from enum import Enum
import os
import requests

from media_bias_insert import index_data_by_url, get_domain_of_url

"""
API Server handling data storage and 'similar claim detection' for project40s.
Note that this file should not be made public while it holds private API keys.
Private API keys could instead be stored in environment variables on Railway if this repo must be made public.
"""
app = Flask(__name__)

# Zilliz Cloud cluster vector database
URI = os.environ['ZILLIZ_URI']
TOKEN = os.environ['ZILLIZ_TOKEN']
# Set up the name of the collection to be created.
COLLECTION_NAME = 'claims'
# Set up the dimension of the embeddings.
DIMENSION = 1536

# OpenAI embedding API
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
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
    UNKNOWN = 'UNKNOWN'


class Fact(DB.Model):
    """ Table storing 'fact-checks' the user has triggered when using the extension. """
    __tablename__ = 'fact'
    # Composite key: 'id of user', 'id of claim', 'url', and 'triggering_text'
    user_id = DB.Column(DB.Text, primary_key=True)
    claim_id = DB.Column(DB.Integer, primary_key=True)
    url = DB.Column(DB.Text, primary_key=True)
    triggering_text = DB.Column(DB.Text, primary_key=True)

    earliest_date_triggered = DB.Column(DB.BigInteger, nullable=False)

    # earliest_of_claim_id = column_property(
    #     select(func.min(earliest_date_triggered)).where(user_id, claim_id).scalar_subquery()
    # )


class Interaction(DB.Model):
    """ Table storing user's media consumption. """
    __tablename__ = 'interaction'
    # Composite key: 'id of user' and 'url'
    user_id = DB.Column(DB.Text, primary_key=True)
    url = DB.Column(DB.String(255), primary_key=True)
    date_spent = DB.Column(DB.BigInteger, primary_key=True)

    duration_spent = DB.Column(DB.Integer, nullable=False)
    clicks = DB.Column(DB.Integer, nullable=False)

    @hybrid_property
    def domain_of_url(self):
        return get_domain_of_url(self.url)


class PoliticalLeaning(DB.Model):
    """ Table storing political leanings of various websites, sourced from TODO """
    __tablename__ = 'political_leaning'

    # Primary key: 'url'
    url = DB.Column(DB.String(255), primary_key=True)

    leaning = DB.Column(DB.Enum(PoliticalLeaningEnum), nullable=False)


# Intialise the database
with app.app_context():
    DB.create_all()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Database - API Routes and helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

    # Renaming the function name (prevents bug - see: https://tinyurl.com/2b35tnvm)
    wrapper.__name__ = func.__name__
    return wrapper


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


@app.route("/recreate", methods=["POST"])
@handle_invalid_request
def recreate_tables():
    request_data = request.get_json()
    check_password(request_data)

    DB.drop_all()
    DB.create_all()

    return None


def get_user_id(token: str) -> str:
    url = 'https://www.googleapis.com/oauth2/v2/userinfo'
    headers = {
        'Authorization': 'Bearer ' + token,
        'Content-Type': 'application/json'
    }

    response = requests.get(url, headers=headers)
    if not response.ok:
        raise InvalidRequest(f'Google OAuth API responded with code "{response.status_code}".')

    return response.json()['id']


def add_facts(user_id, data) -> int:
    num_new_facts = 0

    with rollback_on_err():
        for entry in data:
            new_data = Fact(
                user_id=user_id,
                claim_id=int(get_or_throw(entry, 'claim_id')),
                url=get_or_throw(entry, 'url'),
                triggering_text=get_or_throw(entry, 'triggering_text'),
                earliest_date_triggered=int(get_or_throw(entry, 'earliest_date_triggered')),
            )

            # Check if fact has been triggered before.
            is_new_fact = DB.session.execute(DB.select(Fact).where(
                and_(
                    Fact.user_id == new_data.user_id,
                    Fact.claim_id == new_data.claim_id,
                )
            )).first() is None
            if is_new_fact:
                num_new_facts += 1

            # Add fact to the database
            result = DB.session.execute(DB.select(Fact).where(
                and_(
                    Fact.user_id == new_data.user_id,
                    Fact.claim_id == new_data.claim_id,
                    Fact.url == new_data.url,
                    Fact.triggering_text == new_data.triggering_text
                )
            )).one_or_none()

            if result is None:
                DB.session.add(new_data)
            else:
                # Update entry to the one with the earliest date.
                if new_data.earliest_date_triggered < result.Fact.earliest_date_triggered:
                    result.Fact.earliest_date_triggered = new_data.earliest_date_triggered

    return num_new_facts


@app.route("/facts/add", methods=["POST"])
@handle_invalid_request
def add_facts_endpoint():
    request_data = request.get_json()
    user_id = get_user_id(get_or_throw(request_data, 'oauth_token'))
    url_of_trigger = get_or_throw(request_data, 'url_of_trigger')
    search_results = search(get_or_throw(request_data, 'data'),
                            get_or_throw(request_data, 'similarity_threshold'))['data']

    data = []
    for claim_with_results in search_results:
        for result in claim_with_results['responses']:
            data.append({
                'claim_id': result['id'],
                'url': url_of_trigger,
                'triggering_text': claim_with_results['claim'],
                'earliest_date_triggered': round(time.time() * 1000),
            })
    return add_facts(user_id, data)


@app.route("/facts/get", methods=["POST"])
@handle_invalid_request
def get_all_facts():
    request_data = request.get_json()
    user_id = get_user_id(get_or_throw(request_data, 'oauth_token'))
    after_date = int(request_data.get('after_date', 0))

    with rollback_on_err():
        fact_alias_1 = aliased(Fact)
        fact_alias_2 = aliased(Fact)

        # Get a column representing the first time the fact was seen by the user.
        first_time_seen = DB.select(func.min(fact_alias_2.earliest_date_triggered)).where(
            and_(
                fact_alias_1.user_id == fact_alias_2.user_id,
                fact_alias_1.claim_id == fact_alias_2.claim_id,
            )
        ).label('earliest')

        # Return results, with the most recently seen facts first.
        rows = DB.session.execute(
            DB.select(fact_alias_1, first_time_seen)
            .where(and_(fact_alias_1.user_id == user_id, first_time_seen >= after_date))
            .order_by(fact_alias_1.earliest_date_triggered.desc())
            .order_by(first_time_seen.desc())
        ).all()

        # Index results by claim id. Facts most recently detected will be inserted first.
        results = {}
        for row in rows:
            claim_id = row[0].claim_id
            if claim_id not in results:
                results[claim_id] = []
            results[claim_id] += [{
                'text': row[0].triggering_text,
                'url': row[0].url,
                'earliest_trigger_date': row[0].earliest_date_triggered
            }]

    # Get other fields of claim results
    claim_fields = {
        entry['id']: entry
        for entry in MilvusClient(uri=URI, token=TOKEN).get(
            collection_name=COLLECTION_NAME,
            ids=list(results.keys()),
            output_fields=['id', 'claim', 'author_name', 'author_url', 'review', 'url'],
        )
    }

    # JSON objects don't preserve order, so store in list
    return [{
        'id': claim_id,
        'triggers': triggers,
        'claim': claim_fields[claim_id]['claim'],
        'author_name': claim_fields[claim_id]['author_name'],
        'author_url': claim_fields[claim_id]['author_url'],
        'review': claim_fields[claim_id]['review'],
        'url': claim_fields[claim_id]['url'],
    } for claim_id, triggers in results.items()]


# TODO handle (by returning InvalidRequest):
#   -  fields could not be converted to an int
@app.route("/user_interaction/add", methods=["POST"])
@handle_invalid_request
def add_user_interaction_data():
    request_data = request.get_json()

    new_data = Interaction(
        user_id=get_user_id(get_or_throw(request_data, 'oauth_token')),
        url=get_or_throw(request_data, 'url'),
        duration_spent=int(get_or_throw(request_data, 'duration_spent')),
        date_spent=int(get_or_throw(request_data, 'date_spent')),
        clicks=int(get_or_throw(request_data, 'clicks')),
    )

    with rollback_on_err():
        DB.session.add(new_data)


# TODO handle (by returning InvalidRequest):
#   -  fields could not be converted to an int
@app.route("/user_interaction/get", methods=["POST"])
@handle_invalid_request
def get_user_interaction_data():
    request_data = request.get_json()
    user_id = get_user_id(get_or_throw(request_data, 'oauth_token')),

    with rollback_on_err():
        interactions = DB.session.execute(
            DB.select(Interaction, PoliticalLeaning)
            .join_from(Interaction, PoliticalLeaning, Interaction.domain_of_url == PoliticalLeaning.url, isouter=True)
            .where(and_(Interaction.user_id == user_id))
        ).all()

    results = [{
        'url': row[0].url,
        'date': row[0].date_spent,
        'duration': row[0].duration_spent,
        'clicks': row[0].clicks,
        'leaning': row[1].leaning.value if row[1] is not None else None,
    } for row in interactions]

    return results


@app.route("/bias/get_all", methods=["POST"])
@handle_invalid_request
def insert_media_bias_data():
    request_data = request.get_json()
    check_password(request_data)

    with open(INSERT_MEDIA_BIAS_RAW_DATA) as f:
        data = index_data_by_url(f)

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

    return None

# # TODO handle (by returning InvalidRequest):
# #   -  urls is not a list.
# @app.route("/bias/get", methods=["POST"])
# @handle_invalid_request
# def get_media_bias_data():
#     request_data = request.get_json()
#     urls = get_or_throw(request_data, 'urls')
#
#     with rollback_on_err():
#         rows = DB.session.execute(
#             DB.select(PoliticalLeaning)
#             .where(PoliticalLeaning.url.in_(urls))
#         ).all()
#
#     return [{'url': row[0].url, 'leaning': row[0].leaning} for row in rows]


@app.route("/delete", methods=["POST"])
@handle_invalid_request
def delete_users_data():
    request_data = request.get_json()
    user_id = get_user_id(get_or_throw(request_data, 'oauth_token')),

    with rollback_on_err():
        interaction = DB.session.execute(
            DB.select(Interaction)
            .where(and_(Interaction.user_id == user_id))
        ).all()

        fact = DB.session.execute(
            DB.select(Fact)
            .where(and_(Fact.user_id == user_id))
        ).all()

        for row in interaction:
            DB.session.delete(row[0])
        for row in fact:
            DB.session.delete(row[0])

    return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fact Checking - API Routes and helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    Although parallel processing with more threads would significantly speed up embedding,
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
        output_fields=['id', 'claim', 'author_name', 'author_url', 'review', 'url'],
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

    # Encode on separate threads and return the results
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
