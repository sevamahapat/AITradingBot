from data_loader import load_trading_csv
from tools import set_dataframe
from agent import create_agent
from dotenv import load_dotenv
load_dotenv()

def main():
    df = load_trading_csv("Trades_sample.csv")
    set_dataframe(df)
    agent = create_agent()

    generate_full_report(agent)
    # print("Ask a question (type 'exit' to quit):")
    # while True:
    #     query = input("\nUser: ")
    #     if query.lower() in ["exit", "quit"]:
    #         break
    #     response = agent.run(query)
    #     print(f"Chatbot: {response}")

def generate_full_report(agent):
    total = agent.run("How much was deposited?")
    most_profitable = agent.run("What was the most profitable trade?")
    expiry = agent.run("What percentage of losses came from options expiring?")
    risk = agent.run("What advice would you give this trader to better manage their risk?")

    deep_thought = agent.run(f"""
Use deep_insight tool with the following inputs:
total_deposits: {total}
most_profitable_trade: {most_profitable}
expiry_loss: {expiry}
risk_advice: {risk}
""")

    print("\nTrader Summary Report")
    print("-" * 40)
    print(f"1. Deposits:\n{total}\n")
    print(f"2. Most Profitable Trade:\n{most_profitable}\n")
    print(f"3. Expired Option Losses:\n{expiry}\n")
    print(f"4. Risk Management Advice:\n{risk}\n")
    print(f"5. Deep Behavioral Insight:\n{deep_thought}\n")


if __name__ == "__main__":
    main()
