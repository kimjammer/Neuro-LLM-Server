# main.py
import base64
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, Response
from sse_starlette.sse import EventSourceResponse
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
model.eval()

class ImageURL(BaseModel):
    url: str = ""

class Content(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageURL | None = None

class Message(BaseModel):
    role: str
    content: list[Content]

class ChatRequest(BaseModel):
    messages: list[Message]

class Delta(BaseModel):
    role: str = "assistant"
    content: str = ""

class Choice(BaseModel):
    index: int = 0
    finish_reason: str | None = None
    delta: Delta

class ChatResponse(BaseModel):
    id: str = "chatcmpl-00000"
    object: str = "chat.completions.chunk"
    created: int = 0
    model: str = "MiniCPM-Llama3-V-2_5-int4"
    choices: list[Choice]

app = FastAPI()

def base64_to_image(base64_string):
    # Remove the data URI prefix if present
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]

    # Decode the Base64 string into bytes
    image_bytes = base64.b64decode(base64_string)
    return image_bytes

def create_image_from_bytes(image_bytes):
    # Create a BytesIO object to handle the image data
    image_stream = BytesIO(image_bytes)

    # Open the image using Pillow (PIL)
    image = Image.open(image_stream)
    return image

async def chat_generator(chatRequest: ChatRequest):
    image = None
    msgs = []

    for message in chatRequest.messages:
        for content in message.content:
            if content.type == "text":
                msgs.append({'role': message.role, 'content': content.text})
            elif content.type == "image_url":
                image_bytes = base64_to_image(content.image_url.url)
                image = create_image_from_bytes(image_bytes).convert('RGB')

    ## if you want to use streaming, please make sure sampling=True and stream=True
    ## the model.chat will return a generator
    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
        stream=True
    )

    generated_text = ""
    index = 0
    for new_text in res:
        generated_text += new_text
        print(new_text, flush=True, end='')
        delta = Delta(role="assistant", content=new_text)
        choice = Choice(index=index, finish_reason=None, delta=delta)
        chatResponse = ChatResponse(choices=[choice])
        index += 1
        yield chatResponse.model_dump_json()
    delta = Delta(role="assistant", content="")
    choice = Choice(index=index, finish_reason="stop", delta=delta)
    chatResponse = ChatResponse(choices=[choice])
    yield chatResponse.model_dump_json()


@app.post("/v1/chat/completions")
def chat_completions(chatRequest: ChatRequest):
    return EventSourceResponse(chat_generator(chatRequest))
