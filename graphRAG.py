



from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import os



from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
import os

model_name = "gpt-4o"
deployment = "azure-gpt-4o"

endpoint = os.getenv("AZURE_OPENAI_URL")
subscription_key = os.getenv("AZURE_OPENAI_KEY")
api_version = "2024-12-01-preview"

llm = AzureChatOpenAI(
    azure_deployment=deployment,
    azure_endpoint=endpoint,
    api_version=api_version,  
    temperature=0.7,
    api_key=subscription_key
)
import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential  # Add this import
from dotenv import load_dotenv
load_dotenv()
model_name = "text-embedding-3-small"
deployment = "text-embedding-3-small"

api_version = "2024-02-01"
endpoint = os.getenv("AZURE_OPENAI_URL")
subscription_key = os.getenv("AZURE_OPENAI_KEY")
embedding_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=subscription_key
)

class RealEstateRAG:
    def __init__(self,llm,embedding_model ,openai_api_key=None):
        # Set up OpenAI API key
        #self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        #if not self.api_key:
        #    raise ValueError("OpenAI API key is required")

        # Initialize Neo4j connection
        self.graph = Neo4jGraph(
            url="bolt://localhost:7687",
            username="neo4j",
            password="1234neo4j"
        )

        # Initialize embeddings
        self.embeddings = embedding_model #OpenAIEmbeddings(api_key=self.api_key)

        # Initialize vector store
        # self.vector_store = Neo4jVector.from_existing_graph(
        #     embedding=self.embeddings,
        #     url="bolt://localhost:7687",
        #     username="neo4j",
        #     password="1234neo4j",
        #     search_type="hybrid",  # Combines vector + keyword search
        #     node_label="Building",  # Main node label to search
        #     text_node_properties=["description"],  # Property containing text to embed
        #     embedding_node_property="embedding"  # Property to store embeddings
        # )
        self.llm = llm
        # Initialize LLM
        # self.llm = ChatOpenAI(
        #     temperature=0,
        #     model="gpt-4",
        #     api_key=self.api_key
        # )

        # Set up the Cypher QA Chain
        self.qa_chain = self._setup_qa_chain()

    def _setup_qa_chain(self):
        # Custom prompt template for real estate queries
        CYPHER_PROMPT = """Task: Generate a Cypher query to find real estate information based on the question.

        Database Schema:
        - (Building) nodes with properties: name, description, latitude, longitude, about, in_a_nutshell, etc.
        - (Area) nodes with properties: name, description, about_, in_a_nutshell, community_overview, etc.
        - Specific amenity nodes:
        * (Transport) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (Gym) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (SPA) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (Supermarket) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (Mall) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (MetroStation) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total,duration_mins, distance_km,route_info
        * (Hospital) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (School) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total,duration_mins, distance_km,route_info
        * (Worship) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (Restaurant) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (Park) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (TouristAttraction) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        
        - Relationships: 
        * (Area)-[:CONTAINS]->(Building)
        * (Building)-[:HAS_TRANSPORT]->(Transport)
        * (Building)-[:HAS_GYM]->(Gym)
        * (Building)-[:HAS_SPA]->(SPA)
        * (Building)-[:HAS_SUPERMARKET]->(Supermarket)
        * (Building)-[:HAS_MALL]->(Mall)
        * (Building)-[:HAS_METRO]->(MetroStation)
        * (Building)-[:HAS_HOSPITAL]->(Hospital)
        * (Building)-[:HAS_SCHOOL]->(School)
        * (Building)-[:HAS_WORSHIP]->(Worship)
        * (Building)-[:HAS_RESTAURANT]->(Restaurant)
        * (Building)-[:HAS_PARK]->(Park)
        * (Building)-[:HAS_TOURIST_ATTRACTION]->(TouristAttraction)

        Question: {question}

        Important:
        - Consider spatial relationships when relevant
        - Include ratings and review counts when available
        - Limit results to reasonable numbers (5-10)
        - Filter for operational amenities using business_status='OPERATIONAL' when appropriate
        - Only generate the Cypher query, no other text

        Cypher Query:"""

        cypher_prompt = PromptTemplate(
            template=CYPHER_PROMPT,
            input_variables=["question"]
        )

        return GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            cypher_prompt=cypher_prompt,
            verbose=True,
            allow_dangerous_requests=True
        )
    def answer_query(self, query: str) -> str:
        """
        Process and answer real estate queries using graph capabilities
        """
        try:
            # Use the QA chain to get the answer directly
            # No vector store needed
            response = self.qa_chain.run(query)
            return response

        except Exception as e:
            return f"Error processing query: {str(e)}"
    def answer_query___(self, query: str) -> str:
        """
        Process and answer real estate queries using both vector and graph capabilities
        """
        try:
            # First, try to get relevant context using vector similarity
            vector_results = self.vector_store.similarity_search(
                query,
                k=3  # Get top 3 most relevant results
            )

            # Format vector results for context
            vector_context = "\n".join([doc.page_content for doc in vector_results])

            # Enhance the original query with vector context
            enhanced_query = f"""
            Consider this relevant context:
            {vector_context}

            Original question: {query}
            """

            # Use the QA chain to get the final answer
            response = self.qa_chain.run(enhanced_query)
            return response

        except Exception as e:
            return f"Error processing query: {str(e)}"