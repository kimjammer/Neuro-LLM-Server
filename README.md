# Neuro LLM Server
 
## READ THIS:

This is an extremely jank and hacky implementation of an OpenAI API server for serving the MiniCPM-Llama3-V 2.5 LLM.
I just wanted a straightforward way to use this model over an OpenAI API compliant endpoint, so I hacked this thing together.
Most parameters don't even do anything, it's barely enough to send queries and recieve answers.

## Info

[MiniCPM-Llama3-V 2.5](https://github.com/OpenBMB/MiniCPM-V) is a relatively small but very high quality multimodal llm.
It's finally something small and good enough to use for Neuro! This program will download and use the int4 quantized version
which takes up approximately 8GB of VRAM.

Right now, the server is only set up to do streaming response. I may (or may not) add a few more features and make the endpoint more
compliant to the OpenAI spec.

## Installation

Just don't, but if you really want to...

Only tested on linux. I want to get Windows working too at some point, but it doesn't work right now.
```bash
conda create -n MiniCPM-V python=3.10 -y
conda activate MiniCPM-V
pip install -r requirements.txt
fastapi run
```
btw fastapi dev does not work. (it will try to load the LLM twice and you'll probably run out of VRAM.)