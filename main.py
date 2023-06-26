import requests
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = 'model/'
tokenizer_path = 'tokenizer/'

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

def get_user_repositories_data(github_username):
    url = f"https://api.github.com/users/{github_username}/repos"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    data = json.loads(response.content)
    repositories = [repo["name"] for repo in data]
    return repositories

def preprocess_data(code, tokenizer):
    tokens = tokenizer.encode(code, truncation=True, max_length=512)
    return tokens

def evaluate_code(code, model, tokenizer):
    inputs = tokenizer.encode(code, truncation=True, max_length=512, return_tensors='pt')
    outputs = model.generate(inputs)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

def identify_of_repo(repositories, model, tokenizer, git_username):
    scores = []
    repository_directory = f"https://github.com/{git_username}/"
    for repository in repositories:
        file_path = os.path.join(repository_directory, repository)
        response = requests.get(file_path)
        if response.status_code == 200:
            data = response.text
            preprocessed_code = preprocess_data(data, tokenizer)
            complexity_score = evaluate_code(preprocessed_code, model, tokenizer)
            scores.append((complexity_score, repository))

    if not scores:
        return None
    return max(scores)[1]

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    github_username = data['github_username']
    repositories = get_user_repositories_data(github_username)

    with ThreadPoolExecutor() as executor:
        future = executor.submit(identify_of_repo, repositories, model, tokenizer, github_username)
        most_complex_repository = future.result()

    if most_complex_repository is None:
        response = {"message": "No technically complex repositories found."}
    else:
        response = {"most_complex_repository": most_complex_repository}
    return jsonify(response)

if __name__ == "__main__":
    app.config['TIMEOUT'] = 300
    app.run()
