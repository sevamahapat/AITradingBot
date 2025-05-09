from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from tools import get_total_deposits, get_most_profitable_trade, calculate_expiry_loss_percentage, get_risk_advice, deep_insight

def create_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    tools = [
        get_total_deposits,
        get_most_profitable_trade,
        calculate_expiry_loss_percentage,
        get_risk_advice,
        deep_insight
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent
