import os
from langchain_openai import ChatOpenAI
import logging
from graphreader_agentic_rag.chains.prompts import chunk_read_prompt
from graphreader_agentic_rag.chains.data_models import ChunkOutput, OverallState
from typing import List, Dict
from graphreader_agentic_rag.src.utils import parse_function
from graphreader_agentic_rag.chains.neo4j_utils import neo4j_graph
from graphreader_agentic_rag.chains.initial_node_selection import get_potential_nodes

model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
model = ChatOpenAI(model=model_name, temperature=0.2)

chunk_read_chain = chunk_read_prompt | model.with_structured_output(ChunkOutput)

def get_subsequent_chunk_id(chunk):
    data = neo4j_graph.query("""
    MATCH (c:Chunk)-[:NEXT]->(next)
    WHERE c.id = $id
    RETURN next.id AS next
    """)
    return data

def get_previous_chunk_id(chunk):
    data = neo4j_graph.query("""
    MATCH (c:Chunk)<-[:NEXT]-(previous)
    WHERE c.id = $id
    RETURN previous.id AS previous
    """)
    return data

def get_chunk(chunk_id: str) -> List[Dict[str, str]]:
    data = neo4j_graph.query("""
    MATCH (c:Chunk)
    WHERE c.id = $chunk_id
    RETURN c.id AS chunk_id, c.text AS text
    """, params={"chunk_id": chunk_id})
    return data

def chunk_check(state: OverallState) -> OverallState:
    check_chunks_queue = state.get("check_chunks_queue")
    chunk_id = check_chunks_queue.pop()
    logging.info("-" * 20)
    logging.info(f"Interaction {state.get('iteration_count')}")
    logging.info(f"Step: read chunk({chunk_id})")

    chunks_text = get_chunk(chunk_id)
    read_chunk_results = chunk_read_chain.invoke(
        {
            "question": state.get("question"),
            "rational_plan": state.get("rational_plan"),
            "notebook": state.get("notebook"),
            "previous_actions": state.get("previous_actions"),
            "chunk": chunks_text,
        }
    )

    notebook = read_chunk_results.updated_notebook
    logging.info(
        f"Rational for next action after reading chunks: {read_chunk_results.rational_next_move}"
    )
    chosen_action = parse_function(read_chunk_results.chosen_action)
    logging.info(f"Chosen action: {chosen_action}")
    response = {
        "notebook": notebook,
        "chosen_action": chosen_action.get("function_name"),
        "previous_actions": [f"read_chunks({chunk_id})"],
        "iteration_count": state.get("iteration_count")+1
    }
    if chosen_action.get("function_name") == "read_subsequent_chunk":
        subsequent_id = get_subsequent_chunk_id(chunk_id)
        check_chunks_queue.append(subsequent_id)
    elif chosen_action.get("function_name") == "read_previous_chunk":
        previous_id = get_previous_chunk_id(chunk_id)
        check_chunks_queue.append(previous_id)
    elif chosen_action.get("function_name") == "search_more":
        # Go over to next chunk
        # Else explore neighbors
        if not check_chunks_queue:
            response["chosen_action"] = "search_neighbor"
            # Get neighbors/use vector similarity
            logging.info(f"Neighbor rational: {read_chunk_results.rational_next_move}")
            neighbors = get_potential_nodes(
                read_chunk_results.rational_next_move
            )
            response["neighbor_check_queue"] = neighbors

    response["check_chunks_queue"] = check_chunks_queue
    return response