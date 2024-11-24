import asyncio
from datetime import datetime
from hashlib import md5

import os
from typing import List

from langchain_community.graphs import Neo4jGraph
from langchain_text_splitters import TokenTextSplitter,RecursiveCharacterTextSplitter
from graphreader_agentic_rag.chains.construction_chain import construction_chain

def encode_md5(text):
    return md5(text.encode("utf-8")).hexdigest()

import_query = """
MERGE (d:Document {id:$document_name})
WITH d
UNWIND $data AS row
MERGE (c:Chunk {id: row.chunk_id})
SET c.text = row.chunk_text,
    c.index = row.index,
    c.document_name = row.document_name
MERGE (d)-[:HAS_CHUNK]->(c)
WITH c, row
UNWIND row.atomic_facts AS af
MERGE (a:AtomicFact {id: af.id})
SET a.text = af.atomic_fact
MERGE (c)-[:HAS_ATOMIC_FACT]->(a)
WITH c, a, af
UNWIND af.key_elements AS ke
MERGE (k:KeyElement {id: ke})
MERGE (a)-[:HAS_KEY_ELEMENT]->(k)
"""


class Importer:
    """
    The Importer class is responsible for importing documents into a Neo4j graph database.
    It splits the document into chunks and atomic facts, and then imports them into the graph.
    """
    def __init__(self):
        self.graph = Neo4jGraph(refresh_schema=False)

        self.graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        self.graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:AtomicFact) REQUIRE c.id IS UNIQUE")
        self.graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:KeyElement) REQUIRE c.id IS UNIQUE")
        self.graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")

    # Paper used 2k token size
    async def process_document(self, text, document_name, chunk_size=2000, chunk_overlap=200, separator=[]):
        """
        Processes a document by splitting it into chunks and importing them into the graph.

        Args:
            text (str): The full text of the document.
            document_name (str): The name of the document.
            chunk_size (int, optional): The size of each chunk. Defaults to 2000.
            chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.
            separator (list, optional): A list of separators for recursive text splitting. Defaults to [].

        Returns:
            None
        """
        start = datetime.now()
        print(f"Started extraction at: {start}")

        if separator == []:
            text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=separator,
                chunk_size=100,
                chunk_overlap=0,
                length_function=len,
                is_separator_regex=False,
            )

        texts = text_splitter.split_text(text)
        print(f"Total text chunks: {len(texts)}")
        tasks = [
            asyncio.create_task(construction_chain.ainvoke({"input":chunk_text}))
            for index, chunk_text in enumerate(texts)
        ]
        results = await asyncio.gather(*tasks)
        print(f"Finished LLM extraction after: {datetime.now() - start}")
        docs = [el.dict() for el in results]
        for index, doc in enumerate(docs):
            doc['chunk_id'] = encode_md5(texts[index])
            doc['chunk_text'] = texts[index]
            doc['index'] = index
            for af in doc["atomic_facts"]:
                af["id"] = encode_md5(af["atomic_fact"])
        # Import chunks/atomic facts/key elements
        self.graph.query(import_query, 
                params={"data": docs, "document_name": document_name})
        # Create next relationships between chunks
        self.graph.query("""MATCH (c:Chunk)<-[:HAS_CHUNK]-(d:Document)
        WHERE d.id = $document_name
        WITH c ORDER BY c.index WITH collect(c) AS nodes
        UNWIND range(0, size(nodes) -2) AS index
        WITH nodes[index] AS start, nodes[index + 1] AS end
        MERGE (start)-[:NEXT]->(end)
        """,
            params={"document_name":document_name})
        print(f"Finished import at: {datetime.now() - start}")
    


    async def process_single_file(self, filepath: str, filename: str, separator=[]):
        """
        Processes a single file by reading its content and calling process_document.

        Args:
            filepath (str): The path to the directory containing the files.
            filename (str): The name of the file to process.
            separator (list, optional): A list of separators for recursive text splitting. Defaults to [].

        Returns:
            None
        """
        try:
            with open(os.path.join(filepath, filename), "r") as file:
                text = file.read()
                file_name = filename.replace(".txt", "")
                print(f"Processing {file_name}")
                await self.process_document(text, file_name, chunk_size=2000, chunk_overlap=200, separator=separator)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    async def process_all_files(self, filepath: str, separator=[]):
        """
        Processes all files in a directory.

        Args:
            filepath (str): The path to the directory containing the files.
            separator (list, optional): A list of separators for recursive text splitting. Defaults to [].

        Returns:
            None
        """
        files = os.listdir(filepath)
        tasks: List[asyncio.Task] = []
        
        for filename in files:
            task = asyncio.create_task(self.process_single_file(filepath, filename, separator))
            tasks.append(task)
        
        await asyncio.gather(*tasks)

