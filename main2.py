from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging
import warnings
from langchain_core.tools import tool
from typing import List, Dict, Any

# Disable specific LangChain deprecation warnings
warnings.filterwarnings("ignore", message="The class `Neo4jGraph` was deprecated")
warnings.filterwarnings("ignore", message="The method `Chain.run` was deprecated")

# Configure root logger to show only ERROR and above
logging.getLogger().setLevel(logging.ERROR)

# Configure specific loggers to show INFO and above
logging.getLogger("__main__").setLevel(logging.INFO)

# Create a custom logger for your application
app_logger = logging.getLogger("real_estate_app")
app_logger.setLevel(logging.INFO)

# Create console handler and set level to INFO
console_handler = logging.getLogger().handlers[0] if logging.getLogger().handlers else logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('\033[1;37m[%(levelname)s]\033[0m %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger if it's not already there
if console_handler not in app_logger.handlers:
    app_logger.addHandler(console_handler)

# Suppress other libraries' logs
for logger_name in ['neo4j', 'httpx', 'langchain', 'openai']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

load_dotenv()
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
        Important:
        - The graph database ONLY contains information about areas, buildings, and amenities
        - It does NOT contain price information, property listings, or transaction data
        - Always use CONTAINS for string matching (e.g. WHERE b.name CONTAINS '…')
        - Do NOT use exact equality (WHERE b.name = '…')
        - Always use case-insensitive matching by lowercasing both sides.
          For example:
          WHERE toLower(b.name) CONTAINS toLower('Some Building')
        - Only output the Cypher query — no explanatory text
        Database Schema:
        - (Building) nodes with properties: name, latitude, longitude.
        - (Area) nodes with properties: name.
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
    
    def get_building_coordinates(self, building_name):
        """Get coordinates for a specific building"""
        query = f"""
        MATCH (b:Building)
        WHERE b.name CONTAINS '{building_name}'
        RETURN b.name as name, b.latitude as latitude, b.longitude as longitude, 
               b.description as description
        LIMIT 1
        """
        result = self.graph.query(query)
        return result[0] if result else None

    def get_nearby_amenities(self, building_name, amenity_types=None, radius=1000):
        """
        Get nearby amenities for a building
        
        Parameters:
        - building_name: Name of the building
        - amenity_types: List of amenity types to include (e.g., ["School", "Hospital"])
        - radius: Search radius in meters
        
        Returns:
        - List of amenities with coordinates
        """
        # Default to all amenity types if none specified
        if not amenity_types:
            amenity_types = [
                "Transport", "Gym", "SPA", "Supermarket", "Mall", 
                "MetroStation", "Hospital", "School", "Worship", 
                "Restaurant", "Park", "TouristAttraction"
            ]
        
        # Create relationship patterns for each amenity type
        rel_patterns = []
        for amenity_type in amenity_types:
            rel_name = f"HAS_{amenity_type.upper()}"
            if amenity_type == "MetroStation":
                 rel_name = "HAS_METRO"
            elif amenity_type == "TouristAttraction":
                rel_name = "HAS_TOURIST_ATTRACTION"
                
            rel_patterns.append(f"(b)-[:{rel_name}]->(a:{amenity_type})")
        
        # Join patterns with UNION
        union_query = " UNION ".join([
            f"""
            MATCH {pattern}
            WHERE b.name CONTAINS '{building_name}'
            RETURN a.name as name, a.latitude as latitude, a.longitude as longitude, 
                   a.rating as rating, a.vicinity as vicinity, '{amenity_type}' as type
            """
            for pattern, amenity_type in zip(rel_patterns, amenity_types)
        ])
        
        result = self.graph.query(union_query)
        return result
    def create_commute_map(self, building_name, amenity_types=None, show_route=True):
        """
        Create a map showing a building, nearby amenities, and commute routes
        
        Parameters:
        - building_name: Name of the building
        - amenity_types: List of amenity types to include
        - show_route: Whether to show commute routes
        
        Returns:
        - Folium map object
        """
        building = self.get_building_coordinates(building_name)
        if not building:
            app_logger.error(f"Building not found: {building_name}")
            return None
        
        amenities = self.get_nearby_amenities(building_name, amenity_types)
        
        return create_map([building], amenities, show_route)


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

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

model_name = "gpt-4o"
deployment = "azure-gpt-4o"

api_version = "2024-12-01-preview"
endpoint = os.getenv("AZURE_OPENAI_URL")
subscription_key = os.getenv("AZURE_OPENAI_KEY")
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    azure_endpoint=endpoint,
    api_version=api_version,  
    temperature=0.7,
    api_key=subscription_key
)
crew_llm = "azure/azure-gpt-4o"

# Set the environment variables that LiteLLM will look for
import os
os.environ["AZURE_API_KEY"] = subscription_key
os.environ["AZURE_API_BASE"] = endpoint
os.environ["AZURE_API_VERSION"] = api_version
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OPENTELEMETRY_PYTHON_DISABLED"] = "true"
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

# Connect to your SQLite database
db = SQLDatabase.from_uri("sqlite:///real_estate.db")

# Define your database schema and field descriptions as a system prompt
schema_description = """
    Database: real_estate.db (SQLite)
    Table: listings_buy (Listings about proprties ready to buy)
    Columns: 
    - title (TEXT): Property title/description
    - price (TEXT): Listed price
    - type (TEXT): Property type (Apartment, Villa, etc.)
    - beds (REAL): Number of bedrooms
    - studio (TEXT): Whether it's a studio apartment
    - baths (INTEGER): Number of bathrooms
    - area (TEXT): Property area in square feet
    - location (TEXT): Full location string
    - payment (TEXT): Payment plan
    - agent_plan (TEXT): Agent information
    - building_name (TEXT): Name of the building
    - street_name (TEXT): Street name
    - area_name (TEXT): Area/neighborhood name
    - verified_date (TIMESTAMP): Date when listing was verified
    - handover_quater (TEXT): Quarter of handover
    - handover_year (TEXT): Year of handover
    - offplan_type (TEXT): Off-plan property type
    - type_of_sale (TEXT): Initial Sale or Resale

    Table: listings_rent (Listings about proprties ready to rent)
    Columns:
    - title (TEXT): Property title/description
    - price (TEXT): Rental price
    - type (TEXT): Property type (Apartment, Villa, etc.)
    - beds (REAL): Number of bedrooms
    - studio (TEXT): Whether it's a studio apartment
    - baths (INTEGER): Number of bathrooms
    - area (TEXT): Property area in square feet
    - location (TEXT): Full location string
    - agent_plan (TEXT): Agent information
    - building_name (TEXT): Name of the building
    - street_name (TEXT): Street name
    - area_name (TEXT): Area/neighborhood name
    - verified_date (TIMESTAMP): Date when listing was verified

    Table: transactions_buy (Transactions  about properties that were sold)
    Columns:
    - DATE (TIMESTAMP): Transaction date
    - LOCATION (TEXT): Full location string
    - Type (TEXT): Property type
    - Beds (TEXT): Number of bedrooms
    - BUILT-UP AREA (TEXT): Built-up area in square feet
    - FLOOR (TEXT): Floor number
    - BUILT-UP (TEXT): Additional built-up area information
    - PLOT (TEXT): Plot information
    - building_name (TEXT): Name of the building
    - area_name (TEXT): Area/neighborhood name
    - PRICE (TEXT): Transaction price
    - Info (TEXT): Additional information

    Table: transactions_rent Transactions  about properties that were rented)
    Columns:
    - START DATE (TIMESTAMP): Rental start date
    - LOCATION (TEXT): Full location string
    - Price (TEXT): Rental price
    - Type (TEXT): Property type
    - BEDS (TEXT): Number of bedrooms
    - AREA (SQFT) (TEXT): Area in square feet
    - FLOOR (TEXT): Floor number
    - building_name (TEXT): Name of the building
    - area_name (TEXT): Area/neighborhood name
    - DURATION(Months) (TEXT): Rental duration in months
    - Info (TEXT): Additional information (NEW or RENEWAL)
"""

# Initialize the language model with the schema information
#llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Create a custom prompt template that includes the schema
system_template = f"""You are an AI assistant that translates natural language to SQL.
{schema_description}

When translating natural language to SQL:
1. Only use tables and columns that exist in the schema
2. Be precise with column names and table relationships
3. Use appropriate SQL syntax for SQLite
4. Use the LIKE operator instead of '=' for text matching, and perform case-insensitive comparisons by applying LOWER() to both the column and the search term
5. Return only the SQL query without additional explanation unless asked
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])

# Create the SQL toolkit with the custom prompt
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the SQL agent with the custom system prompt
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    system_message=system_template
)

from crewai.tools import tool

# Define the tools using the decorator
@tool
def get_area_info(query: str) -> str:
    """Get information about Dubai areas, buildings and amenities"""
    rag = RealEstateRAG(llm, embedding_client)
    return rag.answer_query(query)

@tool
def query_property_database(query: str) -> str:
    """Query property listings and transaction data"""
    response = agent.invoke(query)
    return response.content if hasattr(response, 'content') else str(response)

# Define the agents using CrewAI
def create_real_estate_crew(query: str):
    # Planning Agent
    planning_agent = Agent(
        role="Planning Agent",
        goal="Break down complex property queries into sub-tasks and coordinate the workflow",
        backstory="You are an expert real estate planning agent for Dubai properties. You create efficient plans to answer user queries.",
        verbose=True,
        llm=crew_llm,
        system_prompt="""You are a planning agent for a Dubai real estate assistant. 
        Break down the user's query into specific sub-tasks and assign them to the correct agent:
        
        1. GraphRAG Agent: ONLY for information about Dubai areas, buildings, amenities, and location features.
           DO NOT assign financial calculations, property prices, or mortgage analysis to this agent.
        
        2. SQL Agent: ONLY for current property listings, prices, and transaction data.
        
        3. Financial Agent: ONLY for investment calculations, mortgage analysis, ROI projections, 
           loan refinancing, equity calculations, and all other financial matters.
        
        4. Synthesis Agent: For final recommendations after all data is collected.
        
        For each task, specify which agent should handle it. Be very specific about task assignments.
        """
        #allow_delegation=True
    )
    
    # Graph RAG Agent
    graph_rag_agent = Agent(
        role="Graph RAG Agent",
        goal="Provide information about Dubai areas, buildings, and amenities",
        backstory="You are a knowledge expert on Dubai neighborhoods, buildings, and local amenities.",
        verbose=True,
        llm=crew_llm,
        tools=[get_area_info]
    )
    
    # SQL Agent
    sql_agent = Agent(
        role="SQL Agent",
        goal="Provide property listings and transaction data",
        backstory="You are a database expert who can find the most relevant property listings and transaction data.",
        verbose=True,
        llm=crew_llm,
        tools=[query_property_database]
    )
    
    # Synthesis Agent
    synthesis_agent = Agent(
        role="Synthesis Agent",
        goal="Synthesize information to provide final recommendations",
        backstory="You are a Dubai real estate advisor who combines area knowledge and property data to provide valuable recommendations.",
        verbose=True,
        llm=crew_llm
    )
    financial_agent = Agent(
        role="Financial Analyst",
        goal="Calculate accurate ROI and financial metrics for Dubai real estate investments",
        backstory="""You are an expert real estate financial analyst with deep knowledge of Dubai's property market.
        You specialize in calculating ROI, rental yields, and investment potential based on property data and market trends.
        You have access to historical transaction data and can make accurate predictions about investment returns.""",
        verbose=True,
        llm=crew_llm,
        # No tools - using system prompt instead
    )
    # Create the tasks
    planning_task = Task(
        description=f"Create a step-by-step plan to answer this query: '{query}'. Each step should specify which agent to call and what specific information to gather.",
        agent=planning_agent,
        expected_output="A detailed plan with steps for gathering information and creating recommendations"
    )
    
    graph_rag_task = Task(
        description=f"Based on the plan, retrieve information about Dubai areas, buildings, and amenities relevant to: '{query}'",
        agent=graph_rag_agent,
        expected_output="Detailed information about relevant Dubai areas, buildings, and nearby amenities"
    )
    
    sql_task = Task(
        description=f"Based on the plan, query the property database for listings and transactions relevant to: '{query}'",
        agent=sql_agent,
        expected_output="Relevant property listings and transaction data from the database"
    )
    # New Financial Analysis Task
    financial_task = Task(
        description="""
                        You are a real estate finacial investment advisor with access to live Dubai property listings, historical sales and rental data, and local amenities information. Your goal is to help the user maximize the value of their current real estate asset or guide them toward a more profitable or lifestyle-improving property decision, all while considering financial constraints like monthly EMI caps. When advising, always consider market appreciation potential, rental yield, bank refinancing options, and the user’s stated lifestyle or financial goals. Tailor recommendations with recent trends in the region and current property values in mind.
                        Based on the property listings and transaction data gathered, analyze the financial potential of the most promising properties.                    
                        Explain your calculations and provide a final investment analysis summary.
                        """,
        agent=financial_agent,
        expected_output="A comprehensive financial analysis with ROI calculations and investment recommendations"
    )
    synthesis_task = Task(
        description=f"Synthesize the area information, property data, and financial analysis to provide final recommendations for: '{query}'",
        agent=synthesis_agent,
        expected_output="Final recommendations highlighting good value properties with explanations"
    )
    
    crew = Crew(
        agents=[planning_agent, graph_rag_agent, sql_agent, financial_agent, synthesis_agent],
        tasks=[planning_task, graph_rag_task, sql_task, financial_task, synthesis_task],
        verbose=True,
        process=Process.sequential,  # Use sequential process for orchestration
        manager_llm=crew_llm
    )
    
    return crew  # Add this return statement
    
def get_property_recommendations(query: str):
    app_logger.info(f"Processing query: {query}")
    crew = create_real_estate_crew(query)
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    query = "I bought a villa for AED 1 million, paid AED 300,000 as down payment, and currently pay AED 5,000 monthly EMI. The property is now worth AED 2 million. I want to either move to a new home or renovate this one, but I don’t want to increase my monthly EMI. What are my best options?"
    recommendation = get_property_recommendations(query)
    print(recommendation)