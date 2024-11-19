import logging
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

neo4j_graph = Neo4jGraph(refresh_schema=False)

neo4j_vector = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    index_name="keyelements",
    node_label="KeyElement",
    text_node_properties=["id"],
    embedding_node_property="embedding",
    retrieval_query="RETURN node.id AS text, score, {} AS metadata"
)

async def result_sources(result):
    result = result.copy()
    chunk_id = []
    for i in range(0,len(result['previous_actions'])):
        chunk_id_tmp = ""
        if 'read_chunk' in result['previous_actions'][i]:
            #print(result['previous_actions'][i])
            chunk_id_tmp = result['previous_actions'][i]
            chunk_id_tmp = chunk_id_tmp.replace('read_chunks(','')
            chunk_id_tmp = chunk_id_tmp.replace(')','')
            chunk_id.append(chunk_id_tmp)

    chunk_ids_string = ", ".join([f"'{id}'" for id in chunk_id])

    query = f"""
    MATCH (n:Chunk)-[]-(d:Document) 
    WHERE n.id in [{chunk_ids_string}] 
    RETURN distinct d.id AS source
    """
    sources = neo4j_graph.query(query=query)
    sources_string = "### Fontes \n\n"

    for source in sources:
        sources_string += f" - {source['source']}\n"

    result['sources'] = sources_string

    # excluir chave previous_actions

    del result['previous_actions']

    return result

