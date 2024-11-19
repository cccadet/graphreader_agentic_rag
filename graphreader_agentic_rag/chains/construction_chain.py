import os
from langchain_openai import ChatOpenAI
import logging
from graphreader_agentic_rag.chains.prompts import construction_prompt
from graphreader_agentic_rag.chains.data_models import Extraction

model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
model = ChatOpenAI(model=model_name, temperature=0.1)

structured_llm = model.with_structured_output(Extraction)

construction_chain = construction_prompt | structured_llm
