from services.dataset_services import DatasetProvider
from services.llm_services import FineTuningProvider
from services.cronjob_services import queue_job
from datetime import datetime
from typing import Dict
import json
import openai

class OpenAIDatasetProvider(DatasetProvider):
    def __init__(self, dataset):
        super().__init__(dataset)

    def format_dataset(self):
        file_name = "dataset_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl"
        with open(file_name, 'w') as f:
            for entry in self.dataset:
                json.dump(entry, f)
                f.write('\n')
        return file_name
    
    def annotate_data_entry(self, example: Dict):
        return {
            "messages": [
                { "role": "user", "content": example.get("message") },
                { "role": "assistant", "content": example.get("response")}
            ]
        }


class OpenAIFineTuningProvider(FineTuningProvider):
    def __init__(self, dataset):
        super().__init__(dataset)

    def start_fine_tuning_job(self, params):
        openai_client = openai.OpenAI()
        openai_client.fine_tunes.create(params)


import config.env 
from services.chat_services import ChatService
from services.dataset_services import process_dataset_generation_indexing

def fine_tuning_worker():
    dataset = process_dataset_generation_indexing()
    dataset_provider = OpenAIDatasetProvider(dataset)
    fine_tuning_provider = OpenAIFineTuningProvider(dataset_provider)
    fine_tuning_provider.fine_tune(dataset)

queue_job(fine_tuning_worker, 60*24)

chat_service = ChatService()
chat_service.chat()

