from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())
HUGGINGFACE_HUB_API_TOKEN = os.getenv('HUGGINGFACE_HUB_API_TOKEN')


# img2text
def img2text(url):
    image_to_text = pipeline('image-to-text', model='Salesforce/blip-image-captioning-base')

    text = image_to_text(url)[0]['generated_text']

    print(text)
    return text


# llm
def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 20 words;

    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=['scenario'])

    print(story)
    return story


# text to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_HUB_API_TOKEN}"}
    payload = {
        "inputs": message,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


scenario = img2text('photo.jpg')
# story = generate_story(scenario)
text2speech(scenario)
