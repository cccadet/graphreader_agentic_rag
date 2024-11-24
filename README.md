# Graphreader

Graphreader is a Python library that allows you to read and write data from a Neo4j database. It is built on top of the Langgraph/LangChain library, which provides a framework for building language models. It is based on Medium article [Implementing GraphReader with Neo4j and LangGraph](https://towardsdatascience.com/implementing-graphreader-with-neo4j-and-langgraph-e4c73826a8b7)




## Getting started

1. Install the library using pip:


    ```bash
    pip install graphreader_agentic_rag
    ```

2. Configure environment variables (.env):

    - OPENAI_API_KEY='your-openai-api-key'
    - OPENAI_MODEL_NAME='gpt-4o-mini'
    - NEO4J_URI="neo4j+s://xxxxxxxx.databases.neo4j.io"
    - NEO4J_USERNAME=neo4j
    - NEO4J_PASSWORD='your-neo4j-password'
    - ITERATION_LIMIT=12
    
Iteration limit is used to control the number of iterations that the agent will perform. This avoids infinite loops. If not specified, the default value is 10 iterations. Langgraph use 25 iterations. If the value higher than 25, it will cause an error.

3. Import data:

To import data, you can use the [import.ipynb](https://github.com/cccadet/graphreader_agentic_rag/blob/main/examples/import.ipynb) Jupyter notebook available in the `examples` folder of the repository. There, three methods for data import are provided:

- **Text:** Import text.  
- **A single file:** Import a specific file.  
- **All files in a folder:** Import all available files in a folder.  

Currently, only files in the `txt` format are supported. This allows you to reprocess your original files using the tool of your choice.


4. Run the agent:

To run the agent, use the [agent.ipynb](https://github.com/cccadet/graphreader_agentic_rag/blob/main/examples/agent.ipynb) Jupyter notebook available in the `examples` folder of the repository. 


## Roadmap

   - [] Memgraph integration
   - [] Customizable prompting for agents







   







## Add your files





## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

from graphreader.chains.neo4j_utils import result_sources

