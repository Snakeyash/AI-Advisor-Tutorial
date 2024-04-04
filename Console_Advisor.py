import yfinance as yf
from huggingface_hub import InferenceClient

# You need to use your own token. In case of issues please watch the tutorial video I created.
# The token in the code below is expired already:
# Tutorial - https://youtu.be/4JXuNoAfm3g?si=hvIj-C8voAXB9EW2
client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token="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")

robot_output = []
user_input = []

ai_instructions = """
        You are a trading and investing expert.
        You consider only the data provided.
        Double check the price data and date before answering the user query.
        Use the below technical for your predictions:
        1 - price action
        2 - Support and resistance
        3 - Volume 
    """


def extract_price_data(ticker):
    # Define the ticker symbol and the period
    tickersymbol = ticker

    # Get the data for the specified period
    tickerprice = yf.download(tickersymbol, period="2y", interval="1d")
    print(tickerprice)
    return tickerprice


def trigger_ai_analysis(tickerprice):
    # AI instruction set to be an expert

    prompt_engineering = f"<s>[INST]<<SYS>> {ai_instructions} <</SYS>>Provided data: {tickerprice}.[/INST]</s>"

    output = client.text_generation(prompt_engineering, max_new_tokens=5000, temperature=0.01, top_k=1, top_p=0.01)
    print(output)
    return output


def convo_analysis(user_data, system_data, prices):
    prev_user_data = ', '.join(user_data[:-1])
    prev_sys_data = ', '.join(system_data)
    prompt_builder = (f"<s>[INST]<<SYS>> {ai_instructions}"
                      f"Price data: {prices}.<</SYS>>[/INST]"
                      f"Previous user inputs: {prev_user_data}. "
                      f"Previous system outputs: {prev_sys_data}."
                      f"New question: {user_data[-1]}</s>")

    output = client.text_generation(prompt_builder, max_new_tokens=5000, temperature=0.01, top_k=1,
                                    top_p=0.01)
    print(output)
    return output


price_data = extract_price_data("NVDA")
robot_output.append(trigger_ai_analysis(price_data))

while True:
    user_query = input("Enter your input: ")  # Prompt the user for input
    user_input.append(user_query)
    robot_output.append(convo_analysis(user_input, robot_output, price_data))
