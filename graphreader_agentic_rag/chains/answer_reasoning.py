import os
from langchain_openai import ChatOpenAI
import logging
from graphreader_agentic_rag.chains.prompts import answer_reasoning_prompt
from graphreader_agentic_rag.chains.data_models import OutputState, OverallState, AnswerReasonOutput

model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
model = ChatOpenAI(model=model_name, temperature=0.2)

answer_reasoning_chain = answer_reasoning_prompt | model.with_structured_output(AnswerReasonOutput)

def answer_reasoning(state: OverallState) -> OutputState:
    logging.info("-" * 20)
    logging.info("Step: Answer Reasoning")
    final_answer = answer_reasoning_chain.invoke(
        {"question": state.get("question"), "notebook": state.get("notebook")}
    )
    return {
        "answer": final_answer.final_answer,
        "analysis": final_answer.analyze,
        "previous_actions": ["answer_reasoning"],
    }

