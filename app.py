from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import warnings
import streamlit as st
import asyncio

warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv(find_dotenv())
HUGGINGFACE_HUB_API_TOKEN = os.getenv('HUGGINGFACE_HUB_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# img2text
def img2text(url):
    image_to_text = pipeline('image-to-text', model='Salesforce/blip-image-captioning-base', use_fast=True)

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
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    prompt = PromptTemplate(template=template, input_variables=['scenario'])

    chain = prompt | llm
    story = chain.invoke({"scenario": scenario}).content
    print(story)
    return story


# text to speech
async def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_HUB_API_TOKEN}"}
    payload = {
        "inputs": message,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


async def main():
    st.set_page_config(page_title="AI Storyteller", page_icon="📚")
    st.header('Turn img into audio story')
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, 'wb') as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        await text2speech(story)

        with st.expander('scenario'):
            st.write(scenario)
        with st.expander('story'):
            st.write(story)

        st.audio('audio.flac')


if __name__ == '__main__':
    asyncio.run(main())
