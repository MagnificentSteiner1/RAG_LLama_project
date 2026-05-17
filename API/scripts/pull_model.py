import requests
import time

OLLAMA_URL = "http://ollama:11434"

MODELS = [
    "llama3",
    "mxbai-embed-large"
]


def wait_for_ollama():
    print("Waiting for Ollama...")

    while True:
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags")

            if response.status_code == 200:
                print("Ollama is ready.")
                return

        except Exception:
            pass

        time.sleep(2)


def pull_model(model_name: str):
    print(f"Pulling model: {model_name}")

    response = requests.post(
        f"{OLLAMA_URL}/api/pull",
        json={"name": model_name},
        stream=True
    )

    for line in response.iter_lines():
        if line:
            print(line.decode())


wait_for_ollama()

for model in MODELS:
    pull_model(model)

print("All models pulled.")
