# LLM Distillation
This repo provides a framework that can be used to seemlessly integrate LLM Distillation in your existing LLM pipelines. LLM distillation is the process of using small, less costly models in parallel to larger models to save on costs. Here is a <a href="https://substack.recursal.ai/p/run-over-120-npcs-in-a-tiny-ai-town">great resource</a> that provides in depth detail of the process. 

# How to use
As an example, the repo contains `openai_distillation.py` that uses this framework to distill OpenAI models. Here's how you can do it yourrself:
- First, add your API keys inside `/env`. This repo uses Qdrant as an embeddings store
- Extend the `LLMProvider` inside `llm_services.py` to work with your LLM
- Extend the `DatasetProvider` inside `dataset_services.py` to format the collected data according to your LLM
- Finally, extend the `FineTuningProvider` inside `llm_services.py` to create relevant fine tuning jobs

# How it works
## The Data Collection Part
- Each text generation request, and the generated response is collected and stored in an embeddings store
- A tag of `indexed: False` is added to each newly created embedding

## The Fine Tuning Part
- Inside a scheduled job, each embedding with tag `indexed: False` is collected
- The data is formatter and is used to fine tune a small model
- Each embedding's tag is updated to `indexed: True`

## The Distillation Part
- On each request, an AI request router is used
- For the user's query, we search it inside the embeddings store
- If an entry above a certain similarity threshold and `indexed: True` is found, the request is routed to the distilled model
- Else, the request fallbacks to a main model (OpenAI, Mixtral...) and the data is collected to be fine tuned later

| <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc5e0f518-c710-4d42-9f7b-4094fa320263_1593x1083.png" /> | 
|:--:| 
| *Distillation architecture as proposed by Recursal.ai * |


# Acknowledgements
This technique was originally proposed by <a href="https://substack.recursal.ai/">Recursal.ai</a> in their <a href="https://substack.recursal.ai/p/run-over-120-npcs-in-a-tiny-ai-town">🏘️ Run over 120+ NPCs, in a tiny AI town with RWKV</a>. This repo extracts, extends, and generalizes their approach to be used in production. 
