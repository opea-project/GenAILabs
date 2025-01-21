
# MCP Web Retriever Microservice

We provide a simple web retriever server and a client based on the Model-Context-Protocol protocol. Please refer to [mcp](https://modelcontextprotocol.io/quickstart/server) for details.

### Build MCP Google Search web retriever image

```bash
cd ../..
docker build --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/web-retrievers-mcp:latest -f comps/mcp_web_retrievers/Dockerfile .
```

### Start the services

```bash
export host_ip=$(hostname -I | awk '{print $1}')

export OLLAMA_MODEL=qwen2.5-coder
export GOOGLE_API_KEY=$GOOGLE_API_KEY
export GOOGLE_CSE_ID=$GOOGLE_CSE_ID
export OLLAMA_ENDPOINT=http://${host_ip}:11434

systemctl stop ollama.service # docker will start ollama instead, make sure there are no port conflicts
cd comps/mcp_web_retrievers
docker compose -f compose_web_retrievers_mcp.yaml up -d # The first time it will take a few minutes to pull the Ollama model
```

### Run the client

```bash
docker exec -it web-retrievers-mcp bash

python mcp_google_search_client.py mcp_google_search_server.py

# Output log will be like following
# USER_AGENT environment variable not set, consider setting it to identify your requests.

# Connected to server with tools: ['get-google-search-answer']

# MCP Client Started!
# Type your queries or 'quit' to exit.

# Query: search some latest sports news
# ...
# ...
```
