import os 
from dotenv import load_dotenv
from clients.llm_client import LLMClient

def main() -> None:
    #load configuration and create an LLM client.
    load_dotenv()

    api_key = os.getenv("OPENAI_KEY")

    if api_key is None:
        raise ValueError(
            #OPENAI_KEY is missing .Check your .env file.
        )
    client = LLMClient(
        api_key=api_key,
        model = "example-model",
    )

    print("LLM client initialized successfully.")
    print("Selected model:",client.model)

    client.close()

    if __name__ == "__main__":
        main()