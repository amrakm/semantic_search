from audioop import add
import os
import streamlit as st
import numpy as np
import jsonlines
import os
import openai
import json


OPENAI_API_KEY = st.secrets['openai_api_key']


openai.api_key = OPENAI_API_KEY
example_answer = '''
Step 1 Install the service wedge
Step 2 Insert an opening tool
Step 3 Slice through the display adhesive
Step 4 Cut through the remaining adhesive
Step 5 Cut the adhesive along the top left of the display.#
Step 6 Continue along the top of the display.
Step 7 Push the tool around the top right corner of the
display.
Step 8 Wheel the tool down along the right side of the
display.
Step 9 Finish pushing the opening tool to the bottom of the
right side of the display.
Step 10 Separate the display
Step 11 Gently twist the plastic card to open the space
between the display and frame, and cut any remaining
adhesive near the corner.
Step 12 Slide the card toward the center of the display, to cut
any remaining adhesive.
Step 13 Put the card into the corner again and let it stay there
to keep the adhesive from resettling.
Step 14 Insert a second card into the gap between the display
and frame in the top left corner.
Step 15 Gently twist the card, slightly increasing the space
between the display and frame.
tep 16 Slide the plastic card toward the center, again
stopping just before the iSight camera.
Step 17 Insert the card back into the top left corner.#
Step 18 With the cards inserted as shown near the corners,
gently twist the cards to increase the gap between
display and case.'''
uploaded_file_id = "file-L7XDGLnDOQagzrxZbDmNcJkm"

# example_context = processed_list_of_dict[66]['text'][:200]

example_context = ''''iMac 27" 2017 Power Supply Replacement - iFixit Repair Guide\niMac 27" 2017 Power Supply Replacement\nIntroduction\nUse this guide to replace a faulty power supply in your iMac Intel 27" Retina 5K Display'''
example_qa = ['''no power in my iMac 27" 2017, how can I fix it?"''',
              f'''it could be a faulty power supply, use the following steps to fxi it \n{example_answer}''']



def card(name, introduction, score):
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title">{name}</h5>
            <p class="card-text">{introduction}</p>
            <h6 class="card-subtitle mb-2 text-muted">matching score = {score}</h6>
        </div>
    </div>
    """, unsafe_allow_html=True)


def add_answer_card(answer):
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title">Answer</h5>
            <p class="card-text">{answer}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def add_subtitle(subtitle):
    st.markdown(f"""
            <h5 class="card-title">{subtitle}</h5>
    """, unsafe_allow_html=True)

st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
""", unsafe_allow_html=True)


st.write("""
# What is your device name and what seem to be the problem with it? 
""")

question = st.text_input("Ask GPT3!", "")


if question != "":

    answer = openai.Answer.create(
                search_model="ada", 
                model= "text-davinci-002", 
                question=question, 
                file=uploaded_file_id, 
                examples_context=example_context, 
                examples=[example_qa], 
                max_rerank=10,
                max_tokens=500,
                stop=["\n", "<|endoftext|>"],
                return_metadata= True
            )

  


    top_answer = answer['answers'][0]
    add_answer_card(top_answer)

    answers_sorted = sorted(answer['selected_documents'], key=lambda x:-x['score'])

    add_subtitle("I found these releated documents:")


    for ans in answers_sorted:
        title = ans['metadata'] if 'metadata' in ans else ''
        card(
            title,
            ans['text'],
            round(ans['score'], 2)
        )
