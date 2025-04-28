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
        - Always use CONTAINS for string matching (e.g. WHERE b.name CONTAINS '…')
        - Do NOT use exact equality (WHERE b.name = '…')
        - Always use case-insensitive matching by lowercasing both sides.
           For example:
           WHERE toLower(b.name) CONTAINS toLower('Some Building')
        - Only output the Cypher query — no explanatory text
        Database Schema:
        - (Building) nodes with properties: name, description, latitude, longitude, about, in_a_nutshell, etc.
        - (Area) nodes with properties: name, description, about_, in_a_nutshell, community_overview, etc.
        - Specific amenity nodes:
        * (Transport) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (Gym) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (SPA) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (Supermarket) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (Mall) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (MetroStation) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (Hospital) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
        * (School) with properties: name, business_status, vicinity, latitude, longitude, rating, user_ratings_total
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


from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent
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


import os
from typing import List, Dict, Any, TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Configure your API keys
#os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Define the state that will be passed between nodes
class AgentState(TypedDict):
    query: str
    plan: List[str]
    current_step: int
    graph_results: Dict[str, Any]
    sql_results: Dict[str, Any]
    final_answer: str
    messages: List[Any]
    _last_agent:str



def planning_agent(state: AgentState) -> AgentState:
    """Agent that breaks down complex property queries into sub-tasks and coordinates the workflow"""
    query = state["query"]
    app_logger.info("\033[1;34m[PLANNING AGENT]\033[0m Planning the approach...")  # Blue color


    if not state["plan"]:
        planning_prompt = f"""
        You are a planning agent for a Dubai real estate assistant. Break down the user's query into specific sub-tasks.
        You have access to:
        1. GraphRAG Agent: Information about Dubai areas, buildings, amenities like parks hospital, and location features
        2. SQL Agent: Current property listings (buy/rent) and historical transaction data
        
        User query: "{query}"
        
        Create a step-by-step plan to answer this query. Each step MUST start with either:
        - "Call GraphRAG Agent:" followed by the specific task
        - "Call SQL Agent:" followed by the specific task
        
        Dont duplicate the tasks make the tasks concise and minimum number of steps to solve the problem.
        Format each step on a new line with a dash (-) prefix.
        """
        
        response = llm.invoke(planning_prompt)
        
        # Parse the response to extract the plan steps
        plan_steps = [step.strip() for step in response.content.split("\n") if step.strip().startswith("-")]
        if not plan_steps:
            plan_steps = [response.content]
        app_logger.info("\033[1;34m[PLANNING AGENT]\033[0m Created plan:")
        for step in plan_steps:
            app_logger.info(f"\033[1;34m  • {step}\033[0m")
        return {
            **state,  # Include all state properties
            "_last_agent": "planning_agent",
            "plan": plan_steps,
            "current_step": 0,
            "messages": state["messages"] + [AIMessage(content=f"I'll help you find good value properties in Dubai. My plan is:\n" + "\n".join(plan_steps))]
        }
    
    # If we're returning to the planner after an agent has run
    else:
        # Check if we have all the information we need
        has_graph_data = bool(state["graph_results"])
        has_sql_data = bool(state["sql_results"])
        all_steps_completed = state["current_step"] >= len(state["plan"])
        
        # Decide what to do next based on available information
        if all_steps_completed and has_graph_data and has_sql_data:
            # We have everything we need, ready for synthesis
            return {
                **state,  # Include all state properties
                "_last_agent": "planning_agent",
                "messages": state["messages"] + [AIMessage(content="I've gathered all the necessary information. Now I'll provide you with recommendations.")]
            }
        elif state["current_step"] < len(state["plan"]):
            # We still have steps to complete
            current_step = state["plan"][state["current_step"]]
            
            # Determine which agent should handle the current step based on explicit prefixes
            if "Call GraphRAG Agent:" in current_step and not has_graph_data:
                # Need area information
                # Prepare context for the GraphRAG agent if SQL data is available
                context_for_graph = ""
                if state["sql_results"]:
                    context_for_graph = f"SQL data context: {state['sql_results'].get('content', '')}"
                
                return {
                    **state,  # Include all state properties
                    "_last_agent": "planning_agent",
                    "context_for_next_agent": context_for_graph,
                    "messages": state["messages"] + [AIMessage(content=f"Now I'll get information about the area: {current_step}")]
                }
            elif "Call SQL Agent:" in current_step and not has_sql_data:
                # Need property data
                # Prepare context for the SQL agent if graph data is available
                context_for_sql = ""
                if state["graph_results"]:
                    context_for_sql = f"Graph data context: {state['graph_results'].get('content', '')}"
                
                return {
                    **state,  # Include all state properties
                    "_last_agent": "planning_agent",
                    "context_for_next_agent": context_for_sql,
                    "messages": state["messages"] + [AIMessage(content=f"Now I'll get property data: {current_step}")]
                }
            else:
                # Move to the next step
                return {
                    **state,
                    "current_step": state["current_step"] + 1,
                    "messages": state["messages"] + [AIMessage(content=f"Moving to the next step: {state['plan'][state['current_step']]}")]
                }
        else:
            # Fallback - get any missing information
            if not has_graph_data:
                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content="I need to get information about the area first.")]
                }
            elif not has_sql_data:
                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content="I need to get property data next.")]
                }
            else:
                # Ready for synthesis
                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content="I've gathered all the necessary information. Now I'll provide you with recommendations.")]
                }

def graph_rag_agent(state: AgentState) -> AgentState:
    """Agent that provides information about Dubai areas, buildings and amenities"""
    query = state["query"]
    current_step = state["plan"][state["current_step"]] if state["current_step"] < len(state["plan"]) else ""
    app_logger.info("\033[1;32m[GRAPH RAG AGENT]\033[0m Retrieving area information...")  # Green color

    # Get context from previous agents if available
    context = state.get("context_for_next_agent", "")
    
    # Extract the specific task from the current step
    task = current_step.replace("Call GraphRAG Agent:", "").strip() if "Call GraphRAG Agent:" in current_step else query
    
    # Create an enhanced query with context
    enhanced_query = f"{task}\n\n{context}" if context else task
    
    # Use the RealEstateRAG class with the enhanced query
    rag = RealEstateRAG(llm, embedding_client)
    response = rag.answer_query(enhanced_query)
    
    app_logger.info(f"\033[1;32m[GRAPH RAG AGENT]\033[0m Completed task: {task}")

    return {
        **state,  # Preserve all state properties
        "_last_agent": "graph_rag_agent",
        "graph_results": {"content": response, "task": task},
        "current_step": state["current_step"] + 1,
        "messages": state["messages"] + [AIMessage(content=f"{response}")]
    }

def sql_agent(state: AgentState) -> AgentState:
    """Agent that provides property listings and transaction data"""
    query = state["query"].lower()
    current_step = state["plan"][state["current_step"]] if state["current_step"] < len(state["plan"]) else ""
    app_logger.info("\033[1;33m[SQL AGENT]\033[0m Querying property database...")
    # Extract the specific task from the current step
    task = current_step.replace("Call SQL Agent:", "").strip() if "Call SQL Agent:" in current_step else query
    
    # Get context from previous agents if available
    context = state.get("context_for_next_agent", "")
    
    # Get area information from GraphRAG agent if available
    area_info = state["graph_results"].get("content", "No area information available yet.")
    
    # Create an enhanced SQL prompt with context from other agents
    sql_prompt = f"""
    You are a Dubai real estate data agent with access to:
    - Current property listings (buy and rent)
    - Historical transaction data for rent and buy transactions
    - Price per square foot statistics
    
    User query: "{query}"
    Current task: "{task}"
    
    {context}
    
    Area information: {area_info}
    
    When searching via area names use LIKE operator to widen the search results
    """
    
    response = agent.invoke(sql_prompt)

    app_logger.info(f"\033[1;33m[SQL AGENT]\033[0m Completed task: {task}")
    return {
        **state,  # Preserve all state properties
        "_last_agent": "sql_agent",
        "sql_results": {"content": response.content if hasattr(response, 'content') else str(response), "task": task},
        "messages": state["messages"] + [AIMessage(content=f"{response.content if hasattr(response, 'content') else str(response)}")]
    }

def synthesis_agent(state: AgentState) -> AgentState:
    """Agent that synthesizes information from other agents to provide final recommendations"""
    query = state["query"]
    graph_results = state["graph_results"].get("content", "")
    graph_task = state["graph_results"].get("task", "")
    sql_results = state["sql_results"].get("content", "")
    sql_task = state["sql_results"].get("task", "")
    app_logger.info("\033[1;35m[SYNTHESIS AGENT]\033[0m Creating final recommendations...")  # Purple color

    

    synthesis_prompt = f"""
    You are a Dubai real estate advisor helping clients find good value properties.
    
    User query: "{query}"
    
    Area information task: {graph_task}
    Area information results: {graph_results}
    
    Property data task: {sql_task}
    Property data results: {sql_results}
    
    Based on this information, provide a end to end answer showing its reason that highlights good value properties
    to buy or rent in Dubai. Include specific recommendations and explain why they represent good value.
    Make sure to address all aspects of the user's original query.
    Give the final answer in the begining and later the analysis and the data sources.Also make the main answer in BOLD
    """
    
    response = llm.invoke(synthesis_prompt)
    app_logger.info("\033[1;35m[SYNTHESIS AGENT]\033[0m Completed final recommendations")

    # Extract building names from the response if they exist
    result_text = response.content

    return {
        **state,
        "final_answer": response.content,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "_last_agent": "synthesis_agent",
    }

# Define the hub router function
def hub_router(state: AgentState) -> Literal["graph_rag_agent", "sql_agent", "synthesis_agent", "planning_agent", "END"]:
    """Routes between the planner (hub) and specialized agents (spokes)"""
    # If we're just starting, go to the planner
    #print("INSIDE HUB_ROUTER")
    #print(len(state["plan"]))
    if len(state["plan"]) == 0: #not state["plan"]:
        return "planning_agent"
    #print("last_agent",state.get("_last_agent"))
    # If we just came from the planner
    if state.get("_last_agent") == "planning_agent":
        current_step = state["plan"][state["current_step"]] if state["current_step"] < len(state["plan"]) else ""
        
        # Check if we're ready for synthesis
        if state["current_step"] >= len(state["plan"]) and state["graph_results"] and state["sql_results"]:
            return "synthesis_agent"
            
        # Determine which agent to call based on the explicit agent prefix in the current step
        if "Call GraphRAG Agent:" in current_step and not state["graph_results"]:
            return "graph_rag_agent"
        elif "Call SQL Agent:" in current_step and not state["sql_results"]:
            return "sql_agent"
        elif not state["graph_results"]:
            return "graph_rag_agent"
        elif not state["sql_results"]:
            return "sql_agent"
        else:
            return "synthesis_agent"
    
    # If we just came from a specialized agent, go back to the planner
    elif state.get("_last_agent") in ["graph_rag_agent", "sql_agent"]:
        return "planning_agent"
    
    return "planning_agent"

# Build the hub-and-spoke graph
def build_hub_spoke_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planning_agent", planning_agent)
    workflow.add_node("graph_rag_agent", graph_rag_agent)
    workflow.add_node("sql_agent", sql_agent)
    workflow.add_node("synthesis_agent", synthesis_agent)
    
    # Add conditional edges using the router
    workflow.add_conditional_edges(
        "planning_agent",
        hub_router,
        {
            "graph_rag_agent": "graph_rag_agent",
            "sql_agent": "sql_agent",
            "synthesis_agent": "synthesis_agent",
            "planning_agent": "planning_agent",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "graph_rag_agent",
        hub_router,
        {
            "planning_agent": "planning_agent"
        }
    )
    
    workflow.add_conditional_edges(
        "sql_agent",
        hub_router,
        {
            "planning_agent": "planning_agent",
        }
    )
    
    workflow.add_edge("synthesis_agent",END)
    
    # Set the entry point
    workflow.set_entry_point("planning_agent")
    
    return workflow.compile()
# Create the agent executor with hub-and-spoke architecture
dubai_property_agent = build_hub_spoke_graph()


# Example usage
def get_property_recommendations(query: str):
    initial_state = {
        "query": query,
        "plan": [],
        "current_step": 0,
        "graph_results": {},
        "sql_results": {},
        "final_answer": "",
        "messages": [HumanMessage(content=query)]
    }
    
    result = dubai_property_agent.invoke(initial_state)
    return result["final_answer"]

# Test the agent
if __name__ == "__main__":
    query = "I'm looking for a 2-bedroom apartment with good rated schools"
    recommendation = get_property_recommendations(query)
    print(recommendation)
