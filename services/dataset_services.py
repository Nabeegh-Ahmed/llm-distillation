from datetime import datetime
from typing import Dict
import datetime
from services.embedding_services import generate_embeddings, store_embeddings, similarity_scroll, update_indexed_status

def process_chat_entry(message: str, response: str):
    store_embeddings(generate_embeddings(message + " " + response), {
        "message": message,
        "response": response,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "indexed": False
    })

    
def process_dataset_generation_indexing():
    search_result = similarity_scroll(must_filters=[("indexed", False)])
    cleaned_dataset = [{"message": entry.payload.get('message'), "response": entry.payload.get('response')} for entry in search_result]
    update_indexed_status()
    return cleaned_dataset


class DatasetProvider:
    def __init__(self, dataset):
        self.dataset = dataset

    def annotate_dataset(self):
        self.dataset = [self.annotate_data_entry(example) for example in self.dataset]
        return self
    
    def format_dataset(self):
        # use self.dataset here
        raise NotImplementedError()

    def annotate_data_entry(self, example: Dict):
        # Annotate a single entry
        raise NotImplementedError()

