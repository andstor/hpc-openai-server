
#!/usr/bin/env python

import asyncio
import logging
from typing import Any, List, Optional
import ngrok
import os
import gc
from threading import Thread
from pydantic import BaseModel, Field
import uvicorn
from transformers import TextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from fastapi import Body, FastAPI
from torch import seed
import time
from accelerate import Accelerator
from contextlib import asynccontextmanager

from fastapi import HTTPException
from transformers import set_seed

from openai.types.chat import ChatCompletion
from openai.types import Completion
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from fastapi import Request
from src.utils import BatchTextIteratorStreamer
#from src.types import Generation

from os import getenv
from dotenv import load_dotenv
from accelerate.utils.memory import clear_device_cache

from starlette.responses import Response
from traceback import print_exception
import sys



#import logging

#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logger = logging.getLogger(__name__)

from loguru import logger
config = {
    "handlers": [
        {"sink": sys.stdout, "format": "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | <g>{level}</g> |  {message}", "level": "INFO"},
        {"sink": sys.stdout, "format": "<y>{time:YYYY-MM-DD HH:mm:ss}</y> | <y>{level}</y> |  {message}", "level": "WARNING"},
        {"sink": sys.stdout, "format": "<r>{time:YYYY-MM-DD HH:mm:ss}</r> | <r>{level}</r> |  {message}", "level": "ERROR"},        
    ]
}
logger.configure(**config)

#load_dotenv()
PORT = int(getenv("PORT", 8000))
NGROK_AUTH_TOKEN = getenv("NGROK_AUTH_TOKEN")
NGROK_DOMAIN = getenv("NGROK_DOMAIN")
BASIC_AUTH = getenv("BASIC_AUTH")


class ServeModelInfoResult(BaseModel):
    """
    Expose model information
    """

    infos: dict


class ServeTokenizeResult(BaseModel):
    """
    Tokenize result model
    """

    tokens: List[str]
    tokens_ids: Optional[List[int]] = None


class ServeDeTokenizeResult(BaseModel):
    """
    DeTokenize result model
    """

    text: str


class ServeForwardResult(BaseModel):
    """
    Forward result model
    """

    output: Any


# ngrok free tier only allows one agent. So we tear down the tunnel on application termination
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Setting up Ngrok Tunnel")
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    listener = await ngrok.forward(
        addr=PORT,
        domain=NGROK_DOMAIN,
        basic_auth=[BASIC_AUTH], # username:password
    )
    logger.info(f"Ingress established at: {listener.url()}")
    yield
    logger.info("Tearing Down Ngrok Tunnel")
    ngrok.disconnect()


app = FastAPI(lifespan=lifespan)


async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        # you probably want some kind of logging here
        etype, value, tb = sys.exc_info()
        print_exception(etype, value, tb)
        return Response("Internal server error", status_code=500)

app.middleware('http')(catch_exceptions_middleware)


accelerator = Accelerator()
tokenizer = None
model = None


def get_tokenizer(name):
    global tokenizer

    if tokenizer is not None:
        if name == tokenizer:
            return tokenizer
        else:
            del tokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer

def get_model(model_name):
    global tokenizer
    global model

    if model is not None:
        if model_name == model:
            return model
        else:
            del model

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(accelerator.device)
    model.eval()

    return model


@app.post("/v1/tokenizations/tokenize")
def tokenize(
    text: str = Body(None, embed=True),
    tokenizer: str = Body(None, embed=True),
    return_ids: bool = Body(False, embed=True),
    ):
    """
    Tokenize the provided input and eventually returns corresponding tokens id: - **text_input**: String to
    tokenize - **return_ids**: Boolean flags indicating if the tokens have to be converted to their integer
    mapping.
    """
    tokenizer = get_tokenizer(tokenizer)
    try:
        tokens_txt = tokenizer.tokenize(text)

        if return_ids:
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens_txt)
            return ServeTokenizeResult(tokens=tokens_txt, tokens_ids=tokens_ids)
        else:
            return ServeTokenizeResult(tokens=tokens_txt)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"model": "", "error": str(e)})


@app.post("/v1/tokenizations/detokenize", response_model=Any)
def detokenize(
        tokens_ids: List[int] = Body(None, embed=True),
        tokenizer: str = Body(None, embed=True),
        skip_special_tokens: bool = Body(False, embed=True),
        cleanup_tokenization_spaces: bool = Body(True, embed=True),
    ):
        """
        Detokenize the provided tokens ids to readable text: - **tokens_ids**: List of tokens ids -
        **skip_special_tokens**: Flag indicating to not try to decode special tokens - **cleanup_tokenization_spaces**:
        Flag indicating to remove all leading/trailing spaces and intermediate ones.
        """
        tokenizer = get_tokenizer(tokenizer)
        try:
            decoded_str = tokenizer.decode(tokens_ids, skip_special_tokens, cleanup_tokenization_spaces)
            return ServeDeTokenizeResult(model="", text=decoded_str)
        except Exception as e:
            raise HTTPException(status_code=500, detail={"model": "", "error": str(e)})

@app.post("/v1/forward", response_model=Any)
async def forward(
    inputs = Body(None, embed=True),
    model = Body(None, embed=True),
    ):
    """
    **inputs**: **attention_mask**: **tokens_type_ids**:
    """

    model = get_model(model)

    inputs = {k: torch.tensor(v).to(accelerator.device) for k, v in inputs.items()}
    # Check we don't have empty string
    if len(inputs) == 0:
        return ServeForwardResult(output=[], attention=[])

    try:
        # Forward through the model
        with torch.no_grad():
            output = model(**inputs)["logits"].tolist()
        return ServeForwardResult(output=output)
    except Exception as e:
        raise HTTPException(500, {"error": str(e)})

@app.post("/v1/completions")#, response_model=types.Generation)
def generate(generation):#: resources.Generation):
    
    kwargs = generation.model_dump()

    model_name = kwargs.pop("model")
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)
    
    input=tokenizer(kwargs.pop("text"), return_tensors="pt")
    input = {k: v.to(accelerator.device) for k, v in input.items()}

    if kwargs["temperature"] > 0.0:
        kwargs["do_sample"] = True
    else:
        kwargs["do_sample"] = False

    kwargs_clean = {k: v for k, v in kwargs.items() if v is not None}
    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.update(**kwargs_clean)

    outputs = model.generate(**input, generation_config=generation_config)
    choices = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    seed = str(torch.seed())
    created = int(time.time())
    id = seed + "-" + str(created)

    response = types.Generation(id=id, choices=choices, created=created, model=generation.model, system_fingerprint=seed).model_dump()
    
    return response


import json
@app.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat_completion(request: Request):
    kwargs = await request.json()
    logger.info("Received POST request to /v1/chat/completions with body: " + json.dumps(kwargs, indent=4))

    seed = torch.seed()
    if "seed" in kwargs:
        seed = kwargs.pop("seed")
        set_seed(seed)

    model_name = kwargs.pop("model")
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)
    
    messages = kwargs.pop("messages")
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True, # TODO: Check this
        return_tensors="pt",
        return_dict=True,
    ).to(accelerator.device)

    if "temperature" in kwargs and kwargs["temperature"] > 0.0:
        kwargs["do_sample"] = True
    else:
        kwargs["do_sample"] = False

    num_completions = kwargs.pop("n", 1)

    if "max_tokens" in kwargs:
        kwargs["max_length"] = kwargs.pop("max_tokens")
    if "max_completion_tokens" in kwargs:
        kwargs["max_new_tokens"] = kwargs.pop("max_completion_tokens")


    kwargs_clean = {k: v for k, v in kwargs.items() if v is not None}
    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.update(**kwargs_clean)
    
    # if exists and not empty
    if "stop" in kwargs_clean:
        if len(kwargs["stop"]) == 0:
            generation_config.stop_strings = None
        elif type(generation_config.stop_strings) == list:
            generation_config.stop_strings = generation_config.stop_strings.extend(kwargs_clean.pop("stop"))
        else:
            generation_config.stop_strings = kwargs_clean.pop("stop")
    
    if num_completions > 1 and generation_config.temperature == 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot generate multiple completions with temperature=0",
        )

    decode_kwargs = {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False
    }

    outputs = []
    for n in range(1, num_completions + 1):
        logger.info(f"Running {n}/{num_completions} chat completions on conversation with {len(messages)} messages.")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=3, **decode_kwargs)
        
        #outputs = model.generate(**inputs,streamer=streamer, generation_config=generation_config)
        generation_kwargs = dict(inputs, streamer=streamer, tokenizer=tokenizer, generation_config=generation_config)
        thread = Thread(target=model.generate, name = "kodfoko", kwargs=generation_kwargs)
        try:
            thread.start()
        except Exception as e:
            logger.warning(f"Thread failed to start: {e}")
            raise HTTPException(
                status_code=500,
                detail="Thread failed to start",
            )

        output = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            output += new_text
        print()
        outputs.append(output)
        thread.join(None)
        if thread.is_alive():
            logger.warning("Thread is still alive. Killing it.")
            thread.kill()
            
            raise HTTPException(
                status_code=500,
                detail="Thread is still alive. Killing it.",
            )   
        
        clear_device_cache(garbage_collection=True)


    created = int(time.time())
    id = str(seed) + "-" + str(created)

    choices=[]
    for i in range(len(outputs)):
        output = outputs[i]
        choices.append(
            Choice(
                finish_reason="length", # TODO: Add more finish reasons
                index=i,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=output,
                    role="assistant"
                ),
            )
        )
    response = ChatCompletion(
        id=id,
        choices=choices,
        created=created,
        model=model_name,
        object="chat.completion",
    )
    
    return response


if __name__ == "__main__":
    
    # argparse
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the FastAPI server')
    parser.add_argument('--port', type=str, help='Port to run the FastAPI server')
    parser.add_argument('--basic_auth', type=str, help='Basic authentication "username:password"')
    parser.add_argument('--ngrok_auth_token', type=str, help='Ngrok token')
    parser.add_argument('--ngrok_domain', type=str, help='Ngrok domain')
    args = parser.parse_args()
    
    # Set all as environment variables
    for arg in vars(args):
        if getattr(args, arg) is not None:
            os.environ[arg.upper()] = getattr(args, arg)

    PORT = int(getenv("PORT", 8000))
    uvicorn.run("main:app", host="127.0.0.1", port=PORT, reload=True)