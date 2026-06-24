import json
import os
import sys
from typing import Any, Dict

import requests


# Use absolute paths so script works when run from other CWDs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')

try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
except Exception as e:
    print(f'Error reading config.json at {CONFIG_PATH}: {e}', file=sys.stderr)
    raise

# Allow overriding the API URL with an environment variable for flexibility
URL = os.environ.get('API_URL', config.get('url', 'http://127.0.0.1:8000'))
output_model_path = os.path.join(BASE_DIR, config.get('output_model_path', 'models'))


def safe_request(method: str, url: str, **kwargs) -> Dict[str, Any]:
    """Perform an HTTP request and return a serializable dict with status and body.

    - If response is JSON, return parsed JSON under 'body'.
    - If response isn't JSON, return the text under 'body'.
    - On request errors, return the exception message.
    """
    result: Dict[str, Any] = {'url': url, 'method': method}
    try:
        resp = requests.request(method, url, timeout=10, **kwargs)
    except requests.RequestException as e:
        result.update({'error': str(e)})
        return result

    result['status_code'] = resp.status_code
    # Try to parse JSON, otherwise return text
    try:
        result['body'] = resp.json()
    except ValueError:
        # Not JSON
        result['body'] = resp.text
    return result


def main():
    endpoints = {
        'prediction': ('post', f'{URL}/prediction', {'json': {'filepath': 'testdata/testdata.csv'}}),
        'scoring': ('get', f'{URL}/scoring', {}),
        'summarystats': ('get', f'{URL}/summarystats', {}),
        'diagnostics': ('get', f'{URL}/diagnostics', {}),
    }

    responses = {}
    for name, (method, url, kwargs) in endpoints.items():
        print(f'Calling {method.upper()} {url} ...')
        responses[name] = safe_request(method, url, **kwargs)

    os.makedirs(output_model_path, exist_ok=True)
    out_file = os.path.join(output_model_path, 'apireturns.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

    print(f'Wrote API responses to: {out_file}')


if __name__ == '__main__':
    main()
