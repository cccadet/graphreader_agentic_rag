import os
from langchain_openai import ChatOpenAI
import logging
from graphreader_agentic_rag.chains.prompts import neighbor_select_prompt
from graphreader_agentic_rag.chains.data_models import NeighborOutput, OverallState
from graphreader_agentic_rag.src.utils import parse_function

model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
model = ChatOpenAI(model=model_name, temperature=0.2)

neighbor_select_chain = neighbor_select_prompt | model.with_structured_output(NeighborOutput)

def neighbor_select(state: OverallState) -> OverallState:
    logging.info("-" * 20)
    logging.info(f"Interaction {state.get('iteration_count')}")
    logging.info(f"Step: neighbor select")
    logging.info(f"Possible candidates: {state.get('neighbor_check_queue')}")
    neighbor_select_results = neighbor_select_chain.invoke(
        {
            "question": state.get("question"),
            "rational_plan": state.get("rational_plan"),
            "notebook": state.get("notebook"),
            "nodes": state.get("neighbor_check_queue"),
            "previous_actions": state.get("previous_actions"),
        }
    )
    logging.info(
        f"Rational for next action after selecting neighbor: {neighbor_select_results.rational_next_move}"
    )
    chosen_action = parse_function(neighbor_select_results.chosen_action)
    logging.info(f"Chosen action: {chosen_action}")
    # Empty neighbor select queue
    response = {
        "chosen_action": chosen_action.get("function_name"),
        "neighbor_check_queue": [],
        "previous_actions": [
            f"neighbor_select({chosen_action.get('arguments', [''])[0] if chosen_action.get('arguments', ['']) else ''})"
        ],
        "iteration_count": state.get("iteration_count")+1
    }
    if chosen_action.get("function_name") == "read_neighbor_node":
        response["check_atomic_facts_queue"] = [
            chosen_action.get("arguments")[0]
        ]
    return response
