# hpc-openai-server

Transfer files to remote host

scp -r -i path/to/your-key.pem path/to/folder user@your-ec2-public-ip:/path/to/destination/directory


Start fastapi server
uvicorn main:app --host 0.0.0.0 --port 8000

Start fastapi server in houp mode
nohup uvicorn main:app --host 0.0.0.0 --port 8000 &


## Requirements

```
pip install -r requirements.txt
```



```
cp .env.template .env
```

Add your ngrok auth token to the newly created .env file.




## Usage

### Server
usage: main.py [-h] [--port PORT] [--basic_auth BASIC_AUTH] [--ngrok_auth_token NGROK_AUTH_TOKEN] [--ngrok_domain NGROK_DOMAIN]

Run the FastAPI server

optional arguments:
  -h, --help            show this help message and exit
  --port PORT           Port to run the FastAPI server
  --basic_auth BASIC_AUTH
                        Basic authentication "username:password"
  --ngrok_auth_token NGROK_AUTH_TOKEN
                        Ngrok token
  --ngrok_domain NGROK_DOMAIN
                        Ngrok domain

Start the server with the following command:
```
python main.py
```

This will start a local server and expose it to the internet using ngrok. The username and password for the server will be printed to the console. You can now access the server from any device with an internet connection by visiting the ngrok URL provided.

### Client

The server is OpenAI comapatible and can be used with the OpenAI client.

```python
from openai import OpenAI
import base64

auth_key = base64.b64encode("username1:password1".encode()).decode()
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="None",
    default_headers={"Authorization": f"Basic {auth_key}"},
)

messages = [ {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "What is 2+2?"}]

res = client.chat.completions.create(
    messages=messages,
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_completion_tokens=50,
    temperature=1,
    seed=422,
    n=1,
    stop=[],
)

print(res.choices[0].message.content)

# >>> 2 + 2 is equal to 4.
```

