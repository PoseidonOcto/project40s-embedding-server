# Private Embedding Server API

## Installation

To install the dependencies, run the following command in the root directory of the project:

```
pip install -r requirements.txt
```

To start the server, run the following command in the root directory of the project:

```
python main.py
```

To update the dependencies, run the following command in the root directory of the project:

```
pip freeze > requirements.txt
```


## Updating media bias data

To update media bias data, download the new file to the data directory and 
run the following command (in bash) after updating the password field:
```
curl -i --request POST "https://project40s-embedding-server-production.up.railway.app/bias" -H "Content-Type: application/json" -d '{"password": "GET_FROM_RAILWAY"}'
```
> Make sure the global variable storing the path to the data file is appropriate.
> Get the password from the environment variable DATABASE_PRIVILEGED_OP_PASSWORD on Railway.

## Recreating tables

If you have updated the schema or simply wish to recreate the database, 
run the following command (in bash) after updating the password field:
```
curl -i --request POST "https://project40s-embedding-server-production.up.railway.app/recreate" -H "Content-Type: application/json" -d '{"password": "GET_FROM_RAILWAY"}'
```

