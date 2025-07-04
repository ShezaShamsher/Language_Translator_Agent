import os 
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel
from openai import AsyncOpenAI

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

client= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url='https://generativelanguage.googleapis.com/v1beta/'
)

model=OpenAIChatCompletionsModel(model='gemini-2.0-flash', openai_client=client)

config = RunConfig(
    model_provider=client,
    model= model,
    tracing_disabled=True
)

agent = Agent(
    name='Translator Agent',
    instructions=f"you're a pro intelligent language translator agent,when user give you a prompt in any basic language to translator in desired language they ask for, you just simply translator in that specific language. ",
    model=OpenAIChatCompletionsModel(model='gemini-2.0-flash', openai_client=client)
)

print("\n Hey! I'm Language Translator Agent, what's your agenda today? \n")

user_prompt = input("Enter your prompt: ")

result = Runner.run_sync(
    agent,
    user_prompt,
    run_config=config
)

print(result.final_output)