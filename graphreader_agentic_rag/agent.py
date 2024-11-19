import os
from typing import Literal
from langgraph.graph import StateGraph, START, END
from graphreader_agentic_rag.chains.rational_chain import rational_plan_node
from graphreader_agentic_rag.chains.initial_node_selection import initial_node_selection
from graphreader_agentic_rag.chains.atomic_fact_check import atomic_fact_check
from graphreader_agentic_rag.chains.chunk_check import chunk_check
from graphreader_agentic_rag.chains.neighbor_select import neighbor_select
from graphreader_agentic_rag.chains.answer_reasoning import answer_reasoning
from graphreader_agentic_rag.chains.data_models import (InputState,
                    OutputState,
                    OverallState
                    )


class Graphreader():

    def __init__(self):
        self.iteration_limit = int(os.getenv('ITERATION_LIMIT', 10))
        self.run()

    def atomic_fact_condition(self,
        state: OverallState,
    ) -> Literal["neighbor_select", "chunk_check", "answer_reasoning"]:
        if state.get("iteration_count") >= self.iteration_limit:
            return "answer_reasoning"
        elif state.get("chosen_action") == "stop_and_read_neighbor":
            return "neighbor_select"
        elif state.get("chosen_action") == "read_chunk":
            return "chunk_check"

    def chunk_condition(self,
        state: OverallState,
    ) -> Literal["answer_reasoning", "chunk_check", "neighbor_select"]:
        if state.get("iteration_count") >= self.iteration_limit:
            return "answer_reasoning"
        elif state.get("chosen_action") == "termination":
            return "answer_reasoning"
        elif state.get("chosen_action") in ["read_subsequent_chunk", "read_previous_chunk", "search_more"]:
            return "chunk_check"
        elif state.get("chosen_action") == "search_neighbor":
            return "neighbor_select"

    def neighbor_condition(self,
            state: OverallState,
        ) -> Literal["answer_reasoning", "atomic_fact_check"]:
            if state.get("iteration_count") >= self.iteration_limit:
                return "answer_reasoning"
            elif state.get("chosen_action") == "termination":
                return "answer_reasoning"
            elif state.get("chosen_action") == "read_neighbor_node":
                return "atomic_fact_check"

    def run(self): 
        langgraph = StateGraph(OverallState, input=InputState, output=OutputState)
        langgraph.add_node(rational_plan_node)
        langgraph.add_node(initial_node_selection)
        langgraph.add_node(atomic_fact_check)
        langgraph.add_node(chunk_check)
        langgraph.add_node(answer_reasoning)
        langgraph.add_node(neighbor_select)

        langgraph.add_edge(START, "rational_plan_node")
        langgraph.add_edge("rational_plan_node", "initial_node_selection")
        langgraph.add_edge("initial_node_selection", "atomic_fact_check")
        langgraph.add_conditional_edges(
            "atomic_fact_check",
            self.atomic_fact_condition,
        )
        langgraph.add_conditional_edges(
            "chunk_check",
            self.chunk_condition,
        )
        langgraph.add_conditional_edges(
            "neighbor_select",
            self.neighbor_condition,
        )
        langgraph.add_edge("answer_reasoning", END)

        self.langgraph = langgraph.compile()
    
        return self.langgraph
        
