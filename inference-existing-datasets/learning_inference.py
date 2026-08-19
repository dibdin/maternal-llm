#connect to openrouter, the client object is the same everytime
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key="sk-or-v1-...")

#how to make a single API call
response = client.chat.completions.create(
    model="meta-llama/llama-4-scout",
    messages=[{"role":"system","content":"you are a doctor."},
    {"role":"user","content":"What is preeclampsia?"}],
    max_tokens=256,
    temperature=0.0
)

#when you have MCQ and need to turn the format into a prompt: turns a dataset row into a prompt where the question is written out and the options are turned from a string into 4 MCQ choices that are on seperate lines
def format_prompt(question, choices):
    options = ""
    for i, choice in enumerate(choices): 
        options += f"{'ABCD'[i]}. {choice}\n"
    return f"{question}\n\n{options}\nAnswer:"

#write get_answer function that gets the answer from the response text (parses through it for the choice letter), then compare to the correct answer that outputs true or false (later this is averaged over many rows to calculate accuracy)
import re
def get_answer(response_text):
    match = re.search(r'\b[A-D]\b', response_text)
    return match.group(1) if match else ""

pred = get_answer(response.choices[0].message.content)
correct = (pred =="A")

import os, re, pandas as pd
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

df = pd.read_csv("data/filtered/mamabench_afrimedqa.csv").head(5)

for _, row in df.iterrows():
    # 1. build prompt (you'll need to parse row["choices"] — you've seen how)
    # 2. call the API
    # 3. print question, prediction, ground truth
    pass