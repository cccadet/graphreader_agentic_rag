import os
from langchain_openai import ChatOpenAI
import logging
from graphreader_agentic_rag.chains.prompts import atomic_fact_check_prompt
from graphreader_agentic_rag.chains.data_models import AtomicFactOutput, OverallState
from typing import List, Dict
from graphreader_agentic_rag.src.utils import parse_function
from graphreader_agentic_rag.chains.neo4j_utils import neo4j_graph

model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
model = ChatOpenAI(model=model_name, temperature=0.2)

atomic_fact_chain = atomic_fact_check_prompt | model.with_structured_output(AtomicFactOutput)

def get_atomic_facts(key_elements: List[str]) -> List[Dict[str, str]]:
    data = neo4j_graph.query("""
    MATCH (k:KeyElement)<-[:HAS_KEY_ELEMENT]-(fact)<-[:HAS_ATOMIC_FACT]-(chunk)
    WHERE k.id IN $key_elements
    RETURN distinct chunk.id AS chunk_id, fact.text AS text
    """, params={"key_elements": key_elements})
    return data

def get_neighbors_by_key_element(key_elements):
    logging.info(f"Key elements: {key_elements}")
    data = neo4j_graph.query("""
    MATCH (k:KeyElement)<-[:HAS_KEY_ELEMENT]-()-[:HAS_KEY_ELEMENT]->(neighbor)
    WHERE k.id IN $key_elements AND NOT neighbor.id IN $key_elements
    WITH neighbor, count(*) AS count
    ORDER BY count DESC LIMIT 50
    RETURN collect(neighbor.id) AS possible_candidates
    """, params={"key_elements":key_elements})
    return data

def atomic_fact_check(state: OverallState) -> OverallState:
    atomic_facts = get_atomic_facts(state.get("check_atomic_facts_queue"))
    logging.info("-" * 20)
    logging.info(f"Interaction {state.get('iteration_count')}")
    logging.info(f"Step: atomic_fact_check")
    logging.info(
        f"Reading atomic facts about: {state.get('check_atomic_facts_queue')}"
    )
    atomic_facts_results = atomic_fact_chain.invoke(
        {
            "question": state.get("question"),
            "rational_plan": state.get("rational_plan"),
            "notebook": state.get("notebook"),
            "previous_actions": state.get("previous_actions"),
            "atomic_facts": atomic_facts,
        }
    )

    notebook = atomic_facts_results.updated_notebook
    logging.info(
        f"Rational for next action after atomic check: {atomic_facts_results.rational_next_action}"
    )
    chosen_action = parse_function(atomic_facts_results.chosen_action)
    logging.info(f"Chosen action: {chosen_action}")
    response = {
        "notebook": notebook,
        "chosen_action": chosen_action.get("function_name"),
        "check_atomic_facts_queue": [],
        "previous_actions": [
            f"atomic_fact_check({state.get('check_atomic_facts_queue')})"
        ],
        "iteration_count": state.get("iteration_count") + 1
    }
    if chosen_action.get("function_name") == "stop_and_read_neighbor":
        neighbors = get_neighbors_by_key_element(
            state.get("check_atomic_facts_queue")
        )
        response["neighbor_check_queue"] = neighbors
    elif chosen_action.get("function_name") == "read_chunk":
        response["check_chunks_queue"] = chosen_action.get("arguments")[0]
    return response
