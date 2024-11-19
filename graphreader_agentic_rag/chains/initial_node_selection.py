import os
from langchain_openai import ChatOpenAI
from graphreader_agentic_rag.chains.prompts import initial_node_prompt
from graphreader_agentic_rag.chains.data_models import InitialNodes, OverallState
from langchain_core.output_parsers import StrOutputParser
from typing import List
from graphreader_agentic_rag.chains.neo4j_utils import neo4j_vector

model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
model = ChatOpenAI(model=model_name, temperature=0.2)

def get_potential_nodes(question: str) -> List[str]:
    data = neo4j_vector.similarity_search(question, k=50)
    return [el.page_content for el in data]

initial_nodes_chain = initial_node_prompt | model.with_structured_output(InitialNodes)

def initial_node_selection(state: OverallState) -> OverallState:
    potential_nodes = get_potential_nodes(state.get("question"))
    initial_nodes = initial_nodes_chain.invoke(
        {
            "question": state.get("question"),
            "rational_plan": state.get("rational_plan"),
            "nodes": potential_nodes,
        }
    )
    # paper uses 5 initial nodes
    check_atomic_facts_queue = [
        el.key_element
        for el in sorted(
            initial_nodes.initial_nodes,
            key=lambda node: node.score,
            reverse=True,
        )
    ][:5]
    return {
        "check_atomic_facts_queue": check_atomic_facts_queue,
        "previous_actions": ["initial_node_selection"],
        "iteration_count": state.get("iteration_count")+1
    }