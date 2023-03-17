from flask import Response
from parser.parser import Snakefile_SplitByRules


def Tokenize(data):
    match data['format']:
        case 'Snakefile':
            tokenized_data = Snakefile_SplitByRules(data['content'])
        case _:
            return Response(f"Tokenize format not supported: {data['format']}", status=400)
    return tokenized_data
