import os
from langchain_openai import ChatOpenAI
import logging
from graphreader_agentic_rag.chains.prompts import rational_prompt
from graphreader_agentic_rag.chains.data_models import InputState, OverallState
from langchain_core.output_parsers import StrOutputParser

model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

model = ChatOpenAI(model=model_name, temperature=0.2)
rational_chain = rational_prompt | model | StrOutputParser()

def rational_plan_node(state: InputState) -> OverallState:
    rational_plan = rational_chain.invoke({"question": state.get("question")})
    logging.info("-" * 20)
    logging.info(f"Step: rational_plan")
    logging.info(f"Rational plan: {rational_plan}")
    return {
        "rational_plan": rational_plan,
        "previous_actions": ["rational_plan"],
        "iteration_count": 0
    }