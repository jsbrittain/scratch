import json
from typing import List
from parser.parser import Snakefile_SplitByRules


def test_Snakefile_SplitByRules():
    # Load test case
    with open('parser/Snakefile', 'r') as file:
        contents = file.read()
    # Tokenise and split by rule
    data = Snakefile_SplitByRules(contents)
    assert type(data) is str
    result = json.loads(data)
    assert type(result['block'] is List)
    assert len(result['block']) == 6  # six blocks
    for block in result['block']:
        assert 'name' in block
        assert 'code' in block
