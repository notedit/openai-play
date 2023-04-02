from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, column
from sqlalchemy import insert

from llama_index import GPTSQLStructStoreIndex, SQLDatabase, Document
from llama_index import download_loader

from llama_index.indices.struct_store import SQLContextContainerBuilder


engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData(bind=engine)


table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16)),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)
metadata_obj.create_all()


sql_database = SQLDatabase(engine, include_tables=["city_stats"])


WikipediaReader = download_loader("WikipediaReader")
wiki_docs = WikipediaReader().load_data(pages=['Toronto', 'Berlin', 'Tokyo'])


index = GPTSQLStructStoreIndex.from_documents(
    wiki_docs,
    sql_database=sql_database,
    table_name="city_stats",
)


stmt = select(
    [column("city_name"), column("population"), column("country")]
).select_from(city_stats_table)

with engine.connect() as connection:
    results = connection.execute(stmt).fetchall()
    print(results)


city_stats_text = (
    "This table gives information regarding the population and country of a given city.\n"
)
context_documents_dict = {"city_stats": [Document(city_stats_text)]}
context_builder = SQLContextContainerBuilder.from_documents(
    context_documents_dict,
    sql_database
)
context_container = context_builder.build_context_container()

# building the index
index = GPTSQLStructStoreIndex.from_documents(
    wiki_docs,
    sql_database=sql_database,
    table_name="city_stats",
    sql_context_container=context_container,
)
