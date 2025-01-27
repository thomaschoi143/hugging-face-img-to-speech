from openai import OpenAI
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about recursion in programming."},
    ],
    max_tokens=50,
    temperature=0.7,
)

print(completion.choices[0].message.content)
