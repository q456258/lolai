from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

class messageTemplate:

    def fact(self) -> list[str]:
        messages = [
            ("system", "You are a League of Legends assistant. Answer only from the provided context."),
            ("user", "Retrieved reference material{context}, question: {question}"),
        ]
        return ChatPromptTemplate.from_messages(messages=messages)

    def factWithHistory(self) -> list[str]:
        messages = [
            ("system", "You are a League of Legends assistant. Answer only from the provided context."),
            MessagesPlaceholder("history"),
            ("user", "Retrieved reference material{context}, question: {question}"),
        ]
        return ChatPromptTemplate.from_messages(messages=messages)