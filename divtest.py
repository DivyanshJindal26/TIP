import requests

# Define your OpenRouter API key
api_key = "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"

# Set the OpenRouter API endpoint
url = "https://openrouter.ai/api/v1/completions"

# Define headers including API key for authorization
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
aa = input('give me an input: ')
while aa:
    # Define the request payload
    data = {
        "model": "openai/gpt-4",  # Specify the model you want to use (OpenAI's GPT-4 in this case)
        "prompt": aa,  # Your input prompt
        "temperature": 0.7  # Adjust temperature for creative vs. deterministic responses
    }

    # Send the POST request to OpenRouter
    response = requests.post(url, headers=headers, json=data)

    # Parse the response JSON
    output = response.json()

    # Print the generated text
    print(output['choices'][0]['text'])
    aa = input('give me an input: ')
