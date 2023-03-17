from flask import Flask
from flask import request
from flask import Response
from flask_cors import CORS
from flask_cors import cross_origin
import events

app = Flask(__name__)
CORS(app)


@app.route('/api/post', methods=['POST'])
@cross_origin()
def post():
    try:
        app.logger.debug(f'POST request: {request.json}')
        match request.json['query']:
            case 'tokenize':
                data = events.Tokenize(request.json['data'])
            case _:
                return Response(f"Query not supported: {request.json['query']}", status=400)
    except TypeError as err:
        print(err)
        return Response('Server error', status=500)
    return data


@app.route('/')
def main() -> str:
    return '<p>Phyloflow flask server is running</p>'
