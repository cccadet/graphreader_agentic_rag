from langchain_core.prompts import ChatPromptTemplate

rational_plan_system = """As an intelligent assistant, your primary objective is to answer the question by gathering
supporting facts from a given article. To facilitate this objective, the first step is to make
a rational plan based on the question. This plan should outline the step-by-step process to
resolve the question and specify the key information required to formulate a comprehensive answer.
Example:
#####
User: Who had a longer tennis career, Danny or Alice?
Assistant: In order to answer this question, we first need to find the length of Danny’s
and Alice’s tennis careers, such as the start and retirement of their careers, and then compare the
two.
#####
Please strictly follow the above format. Let’s begin."""

rational_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            rational_plan_system,
        ),
        (
            "human",
            (
                "{question}"
            ),
        ),
    ]
)

initial_node_system = """
As an intelligent assistant, your primary objective is to answer questions based on information
contained within a text. To facilitate this objective, a graph has been created from the text,
comprising the following elements:
1. Text Chunks: Chunks of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic
facts derived from different text chunks.
Your current task is to check a list of nodes, with the objective of selecting the most relevant initial nodes from the graph to efficiently answer the question. You are given the question, the
rational plan, and a list of node key elements. These initial nodes are crucial because they are the
starting point for searching for relevant information.
Requirements:
#####
1. Once you have selected a starting node, assess its relevance to the potential answer by assigning
a score between 0 and 100. A score of 100 implies a high likelihood of relevance to the answer,
whereas a score of 0 suggests minimal relevance.
2. Present each chosen starting node in a separate line, accompanied by its relevance score. Format
each line as follows: Node: [Key Element of Node], Score: [Relevance Score].
3. Please select at least 10 starting nodes, ensuring they are non-repetitive and diverse.
4. In the user’s input, each line constitutes a node. When selecting the starting node, please make
your choice from those provided, and refrain from fabricating your own. The nodes you output
must correspond exactly to the nodes given by the user, with identical wording.
Finally, I emphasize again that you need to select the starting node from the given Nodes, and
it must be consistent with the words of the node you selected. Please strictly follow the above
format. Let’s begin.
"""

initial_node_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            initial_node_system,
        ),
        (
            "human",
            (
                """Question: {question}
Plan: {rational_plan}
Nodes: {nodes}"""
            ),
        ),
    ]
)

atomic_fact_check_system = """As an intelligent assistant, your primary objective is to answer questions based on information
contained within a text. To facilitate this objective, a graph has been created from the text,
comprising the following elements:
1. Text Chunks: Chunks of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic
facts derived from different text chunks.
Your current task is to check a node and its associated atomic facts, with the objective of
determining whether to proceed with reviewing the text chunk corresponding to these atomic facts.
Given the question, the rational plan, previous actions, notebook content, and the current node’s
atomic facts and their corresponding chunk IDs, you have the following Action Options:
#####
1. read_chunk(List[ID]): Choose this action if you believe that a text chunk linked to an atomic
fact may hold the necessary information to answer the question. This will allow you to access
more complete and detailed information.
2. stop_and_read_neighbor(): Choose this action if you ascertain that all text chunks lack valuable
information.
#####
Strategy:
#####
1. Reflect on previous actions and prevent redundant revisiting nodes or chunks.
2. You can choose to read multiple text chunks at the same time.
3. Atomic facts only cover part of the information in the text chunk, so even if you feel that the
atomic facts are slightly relevant to the question, please try to read the text chunk to get more
complete information.
#####
Finally, it is emphasized again that even if the atomic fact is only slightly relevant to the
question, you should still look at the text chunk to avoid missing information. You should only
choose stop_and_read_neighbor() when you are very sure that the given text chunk is irrelevant to
the question. Please strictly follow the above format. Let’s begin.
"""

atomic_fact_check_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            atomic_fact_check_system,
        ),
        (
            "human",
            (
                """Question: {question}
Plan: {rational_plan}
Previous actions: {previous_actions}
Notebook: {notebook}
Atomic facts: {atomic_facts}"""
            ),
        ),
    ]
)

chunk_read_system_prompt = """As an intelligent assistant, your primary objective is to answer questions based on information
within a text. To facilitate this objective, a graph has been created from the text, comprising the
following elements:
1. Text Chunks: Segments of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic
facts derived from different text chunks.
Your current task is to assess a specific text chunk and determine whether the available information
suffices to answer the question. Given the question, rational plan, previous actions, notebook
content, and the current text chunk, you have the following Action Options:
#####
1. search_more(): Choose this action if you think that the essential information necessary to
answer the question is still lacking.
2. read_previous_chunk(): Choose this action if you feel that the previous text chunk contains
valuable information for answering the question.
3. read_subsequent_chunk(): Choose this action if you feel that the subsequent text chunk contains
valuable information for answering the question.
4. termination(): Choose this action if you believe that the information you have currently obtained
is enough to answer the question. This will allow you to summarize the gathered information and
provide a final answer.
#####
Strategy:
#####
1. Reflect on previous actions and prevent redundant revisiting of nodes or chunks.
2. You can only choose one action.
#####
Please strictly follow the above format. Let’s begin
"""

chunk_read_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            chunk_read_system_prompt,
        ),
        (
            "human",
            (
                """Question: {question}
Plan: {rational_plan}
Previous actions: {previous_actions}
Notebook: {notebook}
Chunk: {chunk}"""
            ),
        ),
    ]
)

neighbor_select_system_prompt = """
As an intelligent assistant, your primary objective is to answer questions based on information
within a text. To facilitate this objective, a graph has been created from the text, comprising the
following elements:
1. Text Chunks: Segments of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic
facts derived from different text chunks.
Your current task is to assess all neighboring nodes of the current node, with the objective of 
determining whether to proceed to the next neighboring node. Given the question, rational
plan, previous actions, notebook content, and the neighbors of the current node, you have the
following Action Options:
#####
1. read_neighbor_node(key element of node): Choose this action if you believe that any of the
neighboring nodes may contain information relevant to the question. Note that you should focus
on one neighbor node at a time.
2. termination(): Choose this action if you believe that none of the neighboring nodes possess
information that could answer the question.
#####
Strategy:
#####
1. Reflect on previous actions and prevent redundant revisiting of nodes or chunks.
2. You can only choose one action. This means that you can choose to read only one neighbor
node or choose to terminate.
#####
Please strictly follow the above format. Let’s begin.
"""

neighbor_select_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            neighbor_select_system_prompt,
        ),
        (
            "human",
            (
                """Question: {question}
Plan: {rational_plan}
Previous actions: {previous_actions}
Notebook: {notebook}
Neighbor nodes: {nodes}"""
            ),
        ),
    ]
)

answer_reasoning_system_prompt = """
As an intelligent assistant, your primary objective is to answer questions based on information
within a text. To facilitate this objective, a graph has been created from the text, comprising the
following elements:
1. Text Chunks: Segments of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic
facts derived from different text chunks.
You have now explored multiple paths from various starting nodes on this graph, recording key information for each path in a notebook.
Your task now is to analyze these memories and reason to answer the question.
Strategy:
#####
1. You should first analyze each notebook content before providing a final answer.
2. During the analysis, consider complementary information from other notes and employ a
majority voting strategy to resolve any inconsistencies.
3. When generating the final answer, ensure that you take into account all available information.
#####
Example:
#####
User:
Question: Who had a longer tennis career, Danny or Alice?
Notebook of different exploration paths:
1. We only know that Danny’s tennis career started in 1972 and ended in 1990, but we don’t know
the length of Alice’s career.
2. ......
Assistant:
Analyze:
The summary of search path 1 points out that Danny’s tennis career is 1990-1972=18 years.
Although it does not indicate the length of Alice’s career, the summary of search path 2 finds this
information, that is, the length of Alice’s tennis career is 15 years. Then we can get the final
answer, that is, Danny’s tennis career is longer than Alice’s.
Final answer:
Danny’s tennis career is longer than Alice’s.
#####
Please strictly follow the above format. Let’s begin
"""

answer_reasoning_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            answer_reasoning_system_prompt,
        ),
        (
            "human",
            (
                """Question: {question}
Notebook: {notebook}"""
            ),
        ),
    ]
)

construction_system = """
You are now an intelligent assistant tasked with meticulously extracting both key elements and
atomic facts from a long text.
1. Key Elements: The essential nouns (e.g., characters, times, events, places, numbers), verbs (e.g.,
actions), and adjectives (e.g., states, feelings) that are pivotal to the text’s narrative.
2. Atomic Facts: The smallest, indivisible facts, presented as concise sentences. These include
propositions, theories, existences, concepts, and implicit elements like logic, causality, event
sequences, interpersonal relationships, timelines, etc.
Requirements:
#####
1. Ensure that all identified key elements are reflected within the corresponding atomic facts.
2. You should extract key elements and atomic facts comprehensively, especially those that are
important and potentially query-worthy and do not leave out details.
3. Whenever applicable, replace pronouns with their specific noun counterparts (e.g., change I, He,
She to actual names).
4. Ensure that the key elements and atomic facts you extract are presented in the same language as
the original text (e.g., English or Chinese).
"""

construction_human = """Use the given format to extract information from the 
following input: {input}"""

construction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            construction_system,
        ),
        (
            "human",
            (
                "Use the given format to extract information from the "
                "following input: {input}"
            ),
        ),
    ]
)