from llama_index import download_loader, GPTSimpleVectorIndex, LLMPredictor, ServiceContext, GPTListIndex
from pathlib import Path

from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph


from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig, GraphToolConfig

from llama_index.indices.query.query_transform.base import DecomposeQueryTransform


years = [2022, 2021, 2020, 2019]
UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)

loader = UnstructuredReader()
doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(
        file=Path(f'./data/UBER/UBER_{year}.html'), split_documents=False)
    # insert year metadata into each year
    for d in year_docs:
        d.extra_info = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)


# service_context = ServiceContext.from_defaults(chunk_size_limit=512)
# index_set = {}
# for year in years:
#     cur_index = GPTSimpleVectorIndex.from_documents(
#         doc_set[year], service_context=service_context)
#     index_set[year] = cur_index
#     cur_index.save_to_disk(f'index_{year}.json')


# Load indices from disk
index_set = {}
for year in years:
    cur_index = GPTSimpleVectorIndex.load_from_disk(f'index_{year}.json')
    index_set[year] = cur_index


index_summaries = [
    f"UBER 10-k Filing for {year} fiscal year" for year in years]

# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# define a list index over the vector indices
# allows us to synthesize information across each index
graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index_set[y] for y in years],
    index_summaries=index_summaries,
    service_context=service_context,
)

# [optional] save to disk
graph.save_to_disk('10k_graph.json')
# [optional] load from disk, so you don't need to build graph from scratch
graph = ComposableGraph.load_from_disk(
    '10k_graph.json',
    service_context=service_context,
)


decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True
)

# define query configs for graph
query_configs = [
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 1,
            # "include_summary": True
        },
        "query_transform": decompose_transform
    },
    {
        "index_struct_type": "list",
        "query_mode": "default",
        "query_kwargs": {
            "response_mode": "tree_summarize",
            "verbose": True
        }
    },
]
# graph config
graph_config = GraphToolConfig(
    graph=graph,
    name=f"Graph Index",
    description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber.",
    query_configs=query_configs,
    tool_kwargs={"return_direct": True}
)

index_configs = []
for y in years:
    tool_config = IndexToolConfig(
        index=index_set[y],
        name=f"Vector Index {y}",
        description=f"useful for when you want to answer queries about the {y} SEC 10-K for Uber",
        index_query_kwargs={"similarity_top_k": 3},
        tool_kwargs={"return_direct": True}
    )
    index_configs.append(tool_config)


toolkit = LlamaToolkit(
    index_configs=index_configs,
    graph_configs=[graph_config]
)


memory = ConversationBufferMemory(memory_key="chat_history")
llm = OpenAI(temperature=0)
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True
)


agent_chain.run(
    input="What were some of the biggest risk factors in 2020 for Uber")
