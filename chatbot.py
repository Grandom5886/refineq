import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate

# Define the repo ID and connect to Mixtral model on Huggingface
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(
  repo_id=repo_id, 
  model_kwargs={"temperature": 0.8, "top_k": 50}, 
  huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
)

template = """
You are a fashion store chatbot who helps finding products based on users queries. user will ask you to find the clothes or mentions the occation. 
Use following piece of context to answer the question. 
If you don't know the answer, just say you don't know the answer. 
Keep the answer concise.

Context: {context}
Question: {question}
Answer: 

"""


st.set_page_config(page_title="Ecommerce Chatbot")
with st.sidebar:
    st.title('Mumbai Marines : Ecommerce Product Search')

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = API_KEY
)

def get_openai_context(prompt:str, chat_history:str, max_tokens) -> str:
    """Get context from OpenAI model."""
    response = client.chat.completions.create(
      model="meta/llama-3.1-405b-instruct",
      messages=[
          {"role":"system","content":prompt},
          {"role": "user", "content": chat_history}
      ],
      temperature=1,
      max_tokens=max_tokens
    )
    return response.choices[0].message.content

# Function for generating LLM response
def generate_response(input,max_tokens):
    result = get_openai_context("",input,max_tokens)
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello there!, i am your search assisstant, i can help you find the right products, How can i help you now? 😊"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def extract_details(prompt):
    # Define prompts for LLM to determine gender and occasion
    gender_prompt = f"Identify the gender for this fashion query from [men/women/boy/girl] for whom the user is trying to buy the product for: {prompt}"
    occasion_prompt = f"This query is about a particular type of product: {prompt}"
    gender=generate_response(gender_prompt,1)
    occasion=generate_response(occasion_prompt,1)
    return gender,occasion



def check_context(prompt):
    greet = False
    genderspecific=False
    occasionspecified=False
    gender=""
    occasion=""
    greetings_prompt = f"Identify whether it is a greeting. reply with yes or no : {prompt}"
    greet=generate_response(greetings_prompt,1)
    if greet.lower=="true":
        return generate_response("Reply politely to the query also mention that you are a fashion help assistant who can help with product search and recommend based on occasion for the prompt: {prompt}",50)
   
    
    isgender_prompt = f"This is about internal product search. Does this prompt gives enough context to understand to a specic gender the product search is for? answer with yes or no: {prompt}"
    findingoccasion = f"Does this prompt provide enough context to identify for which occasion they are finding clothes for or they are specific about which product to buy reply with yes or no :{prompt}"

    if generate_response(isgender_prompt,1).lower=="true":
        genderspecific = True
    if generate_response(findingoccasion,1).lower=="true":
        occasionspecified = True
    while not genderspecific or not occasionspecified:
        if not genderspecific:
            response=generate_response("As the gender is not specific in the previous prompt. Ask for the specific gender for which they wish to buy fashion products for.",50)
        elif not occasionspecified:
            response=generate_response("As the occasion is not specific in the previous prompt. Ask for the specific occasion or specific clothes they have in mind for which they wish to buy fashion products for.",50)
        else:
            gender=extract_details(prompt)
    return response
            
# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = check_context(input)
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)