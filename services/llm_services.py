from openai import OpenAI
from typing import List
from services.dataset_services import DatasetProvider

class SupportedModels:
    GPT_3 = "gpt-3.5-turbo"
    PHI_2 = "microsoft/phi-2.0"

class LLMProvider:
    def generate(self, messages):
        raise NotImplementedError()
    
class GPTProvider(LLMProvider):
    class ChatMessageRoles:
        ASSISTANT = "assistant"
        SYSTEM = "system"
        USER = "user"

    class ChatMessage:
        role: 'GPTProvider.ChatMessageRoles'
        content: str

    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model

    def generate(self, messages: List[ChatMessage]):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return completion.choices[0].message.content

def get_model(model_name):
    if model_name == SupportedModels.GPT_3:
        return GPTProvider(model_name)

class FineTuningProvider:
    def __init__(self, dataset_provider: 'DatasetProvider'):
        self.dataset_provider = dataset_provider
    
    def fine_tune(self):
        params = self.dataset_provider.annotate_dataset().format_dataset()
        self.start_fine_tuning_job(params)

    
    def start_fine_tuning_job(self, params):
        raise NotImplementedError()