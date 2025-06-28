from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
import requests
import os
import streamlit as st
import torch
# from bark import generate_audio, preload_models
# from scipy.io.wavfile import write as write_wav
# import numpy as np

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HuggingfaceHubToken")

# img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    print(text)
    return text

# llm
def generate_story(scenario):
    template = """
    You are a story teller and always write in a positive tone;
    You can generate a short story based on a simple narrative, the story should be no more than 200 words with a positive ending;

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    
    story = chain.run(scenario=scenario)
    print(story)
    return story

# text to speech
def text2speech(message):
    # API_URL = "https://router.huggingface.co/fal-ai/fal-ai/chatterbox/text-to-speech"
    # API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
    
    # headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    # payloads = {"inputs": message}
    # response = requests.post(API_URL, headers=headers, json=payloads)
    # print("Status Code:", response.status_code)
    # print("Content-Type:", response.headers.get("Content-Type"))
    # print("First 100 bytes:", response.content[:100])
    
    # # Use Bark locally
    # preload_models()
    # audio_array = generate_audio(message)
    # audio_array = np.int16(audio_array * 32767)
    # write_wav("bark_output.wav", rate=24000, data=audio_array)

    # ElevenLabs
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel (default)
    headers = {
    "xi-api-key": api_key,
    "Content-Type": "application/json"
    }
    json_data = {
        "text": message,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    response = requests.post(url, headers=headers, json=json_data)
    if response.status_code == 200:
        with open("audio.mp3", "wb") as f:
            f.write(response.content)
        print("Saved audio to audio.mp3")
    else:
        print("Error:", response.status_code, response.text)


# run
# scenario = img2text("./photo.jpg")
# story = generate_story(scenario)
# text2speech(story)

def main():

    st.set_page_config(page_title="img 2 audio story", page_icon="🧠")

    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)

        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.mp3", format="audio/mp3")

if __name__ == "__main__":
    main()
