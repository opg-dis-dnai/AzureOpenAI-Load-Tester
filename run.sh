ENDPOINT="http://127.0.0.1:8000/v1"
API_KEY="f28a6dc9d0374ecda426bef1c999924a"
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
CONCURRENCY_LEVEL="20"
TOKEN_COUNT="1000"
DURATION="30s"
MAX_TOKENS="8000"

python main.py --endpoint $ENDPOINT --api-key $API_KEY --model $MODEL --concurrency-level $CONCURRENCY_LEVEL --token-count $TOKEN_COUNT --duration $DURATION --max-tokens $MAX_TOKENS