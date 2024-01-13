from services.llm_services import get_model, SupportedModels
from services.embedding_services import generate_embeddings, similarity_search
from services.dataset_services import process_chat_entry

class ChatService:
    def __init__(self, main_model: 'SupportedModels' = SupportedModels.GPT_3, distilled_model: 'SupportedModels' = SupportedModels.PHI_2, similarity_threshold: float = 0.8):
        self.messages = []
        self.model = get_model(model_name=main_model)
        self.distilled_model = get_model(model_name=distilled_model)
        self.similarity_threshold = similarity_threshold

    def chat(self):
        while True:
            message = input("User: ")
            if message == "exit":
                break
            self.messages.append(self.model.ChatMessage(
                role=self.model.ChatMessageRoles.USER,
                content=message
            ))

            similarity_search_result = similarity_search(
                generate_embeddings(message), 
                must_filters=[("indexed", True)], 
                score_threshold=self.similarity_threshold
            )
            if len(similarity_search_result.result) > 0:
                self.distilled_model.generate(self.messages)
                continue

            response = self.model.generate(self.messages)
            print(f"AI: {response}")

            self.messages.append(self.model.ChatMessage(
                role=self.model.ChatMessageRoles.ASSISTANT,
                content=response
            ))

            process_chat_entry(message, response)