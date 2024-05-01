import os
import json
import pandas as pd
import traceback
from langchain.chat_models import ChatOpenAI
from langchain.llms import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
import PyPDF2


llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


TEMPLETE="""
Text:{text}
you are an expert MCQ generator. given the above text it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. \
make sure questions are not repeated and also make sure your respone looks like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs
###RESPONSE_JSON
{response_json}
"""

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}


prompt = PromptTemplate(input_variables=["text", "number", "subject", "tone", "response_json"],template=TEMPLETE)

chain1 = LLMChain(llm = llm,prompt=prompt,output_key="quiz", verbose=True)

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation=PromptTemplate(input_variables=["subject","quiz"],template=TEMPLATE2)

chain2 = LLMChain(llm=llm,prompt=quiz_evaluation, output_key="review", verbose = True)



seq_chain = SequentialChain(chains=[chain1, chain2], input_variables=["text", "number", "subject", "tone", "response_json"],output_variables=["quiz", "review"], verbose=True)



with open("../psychology.txt", "r") as file:
    text= file.read()
    print(text)


    response= seq_chain({
        "text": text,
        "number" : 5,
        "subject" : "psychology",
        "tone" : "simple",
        "response_json" : json.dumps(RESPONSE_JSON)
    })

print(response.get("quiz"))
print(response.get("review"))

