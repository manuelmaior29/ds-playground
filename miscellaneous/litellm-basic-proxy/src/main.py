import openai

client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000" # Connect to proxy
)

response = client.chat.completions.create(model="plm", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)