import streamlit as st
import os
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
import time
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
import requests
import polyline
import folium
from datetime import datetime, timedelta
import streamlit as st
from streamlit_folium import folium_static

# Import your existing agent components
from main2 import get_property_recommendations
import sys
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
import os
load_dotenv()

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

from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential  # Add this import
from graphRAG import RealEstateRAG
model_name = "text-embedding-3-small"
deployment = "text-embedding-3-small"

api_version = "2024-02-01"
endpoint = os.getenv("AZURE_OPENAI_URL")
subscription_key = os.getenv("AZURE_OPENAI_KEY")
print(subscription_key)
embedding_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=subscription_key
)

def create_map(buildings, amenities, show_route=True):
    """
    Create a folium map showing buildings and nearby amenities with routes
    
    Parameters:
    - buildings: List of building dictionaries with name, latitude, longitude
    - amenities: List of amenity dictionaries with name, type, latitude, longitude
    - show_route: Whether to show routes between buildings and amenities
    
    Returns:
    - Folium map object
    """
    # Create a map centered on the first building
    if buildings and len(buildings) > 0:
        center_lat = buildings[0]['latitude']
        center_lng = buildings[0]['longitude']
    else:
        # Default to Dubai center if no buildings
        center_lat = 25.2048
        center_lng = 55.2708
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=14)
    
    # Add building markers
    for building in buildings:
        if 'latitude' in building and 'longitude' in building:
            popup_content = f"""
            <b>{building['name']}</b><br>
            {building.get('description', '')}
            """
            folium.Marker(
                location=[building['latitude'], building['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='blue', icon='home', prefix='fa'),
                tooltip=building['name']
            ).add_to(m)
    
    # Create marker clusters for amenities
    amenity_clusters = {}
    amenity_colors = {
        'Transport': 'gray',
        'MetroStation': 'red',
        'Supermarket': 'green',
        'Restaurant': 'orange',
        'School': 'purple',
        'Hospital': 'darkred',
        'Park': 'darkgreen',
        'Mall': 'pink',
        'Gym': 'darkblue',
        'SPA': 'lightblue',
        'Worship': 'beige',
        'TouristAttraction': 'cadetblue'
    }
    
    # Create a cluster for each amenity type
    for amenity_type in amenity_colors.keys():
        amenity_clusters[amenity_type] = MarkerCluster(name=amenity_type).add_to(m)
    
    # Add amenity markers to appropriate clusters
    for amenity in amenities:
        if isinstance(amenity, dict) and 'latitude' in amenity and 'longitude' in amenity:
            amenity_type = amenity.get('type', 'Other')
            
            # Format commute information if available
            commute_info = ""
            if 'distance' in amenity and amenity['distance']:
                distance_text = f"{amenity['distance'] / 1000:.1f} km" if isinstance(amenity['distance'], (int, float)) else amenity['distance']
                commute_info += f"<br>Distance: {distance_text}"
            
            if 'duration' in amenity and amenity['duration']:
                duration_mins = amenity['duration'] / 60 if isinstance(amenity['duration'], (int, float)) else amenity['duration']
                if isinstance(duration_mins, (int, float)):
                    duration_text = f"{duration_mins:.0f} mins"
                else:
                    duration_text = duration_mins
                commute_info += f"<br>Travel time: {duration_text}"
            
            popup_content = f"""
            <b>{amenity['name']}</b><br>
            Type: {amenity_type}<br>
            {amenity.get('vicinity', '')}{commute_info}
            """
            
            # Choose the right cluster based on amenity type
            cluster = amenity_clusters.get(amenity_type, amenity_clusters.get('Other', m))
            
            # Choose icon color based on amenity type
            icon_color = amenity_colors.get(amenity_type, 'gray')
            
            # Choose icon based on amenity type
            icon_name = 'info'
            if amenity_type == 'Transport' or amenity_type == 'MetroStation':
                icon_name = 'subway'
            elif amenity_type == 'Supermarket':
                icon_name = 'shopping-cart'
            elif amenity_type == 'Restaurant':
                icon_name = 'utensils'
            elif amenity_type == 'School':
                icon_name = 'graduation-cap'
            elif amenity_type == 'Hospital':
                icon_name = 'hospital'
            elif amenity_type == 'Park':
                icon_name = 'tree'
            elif amenity_type == 'Mall':
                icon_name = 'shopping-bag'
            elif amenity_type == 'Gym':
                icon_name = 'dumbbell'
            
            folium.Marker(
                location=[amenity['latitude'], amenity['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa'),
                tooltip=amenity['name']
            ).add_to(cluster)
            
            # Add route lines if requested and route_info is available
            if show_route and buildings and 'route_info' in amenity and amenity['route_info'] and 'polyline' in amenity['route_info']:
                route_points = amenity['route_info']['polyline']
                if route_points:
                    folium.PolyLine(
                        locations=route_points,
                        color=icon_color,
                        weight=3,
                        opacity=0.7,
                        tooltip=f"Route to {amenity['name']}"
                    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# Add this to your RealEstateRAG class
def get_commute_info(self, building_name, amenity_types=None, mode="driving"):
    """
    Get commute information from a building to nearby amenities
    
    Parameters:
    - building_name: Name of the building
    - amenity_types: List of amenity types to include
    - mode: Travel mode (driving, walking, bicycling, transit)
    
    Returns:
    - List of amenities with commute information
    """
    building = self.get_building_coordinates(building_name)
    if not building:
        app_logger.error(f"Building not found: {building_name}")
        return None
    
    amenities = self.get_nearby_amenities(building_name, amenity_types)
    
    # Import the directions function
    from googlemaps_util import get_directions
    
    # Add commute information to each amenity
    for amenity in amenities:
        if 'latitude' in amenity and 'longitude' in amenity:
            try:
                route_info = get_directions(
                    building['latitude'], building['longitude'],
                    amenity['latitude'], amenity['longitude'],
                    mode=mode
                )
                
                if route_info:
                    amenity['route_info'] = route_info
                    amenity['distance'] = route_info.get('distance', {}).get('value')
                    amenity['duration'] = route_info.get('duration', {}).get('value')
            except Exception as e:
                app_logger.error(f"Error getting directions: {str(e)}")
    
    return amenities

# Add this to your RealEstateRAG class
def create_commute_map(self, building_name, amenity_types=None, mode="driving", show_route=True):
    """
    Create a map showing a building, nearby amenities, and commute routes
    
    Parameters:
    - building_name: Name of the building
    - amenity_types: List of amenity types to include
    - mode: Travel mode (driving, walking, bicycling, transit)
    - show_route: Whether to show commute routes
    
    Returns:
    - Folium map object
    """
    building = self.get_building_coordinates(building_name)
    if not building:
        app_logger.error(f"Building not found: {building_name}")
        return None
    
    amenities = self.get_commute_info(building_name, amenity_types, mode)
    
    return create_map([building], amenities, show_route)

# Add this to your GraphRAG Agent tools
@tool
def get_building_commute_map(query: str) -> str:
    """
    Generate a map showing commute times from a building to nearby amenities.
    The query should include the building name and optionally specific amenity types.
    """
    # Parse the query to extract building name and amenity types
    # This is a simple implementation - you might want to use NLP for better extraction
    building_name = None
    amenity_types = None
    mode = "driving"
    
    # Extract building name - assuming it's mentioned after "from" or similar phrases
    if "from" in query.lower():
        parts = query.lower().split("from")
        if len(parts) > 1:
            building_name_part = parts[1].strip()
            # Take the first few words as the building name
            building_name = ' '.join(building_name_part.split()[:3])
    
    # Extract amenity types if specified
    amenity_keywords = {
        "metro": "MetroStation",
        "transport": "Transport",
        "supermarket": "Supermarket",
        "restaurant": "Restaurant",
        "school": "School",
        "hospital": "Hospital",
        "park": "Park",
        "mall": "Mall",
        "gym": "Gym",
        "spa": "SPA",
        "worship": "Worship",
        "tourist": "TouristAttraction"
    }
    
    requested_amenities = []
    for keyword, amenity_type in amenity_keywords.items():
        if keyword in query.lower():
            requested_amenities.append(amenity_type)
    
    if requested_amenities:
        amenity_types = requested_amenities
    
    # Extract travel mode if specified
    if "walking" in query.lower():
        mode = "walking"
    elif "transit" in query.lower() or "public transport" in query.lower():
        mode = "transit"
    
    if not building_name:
        return "Please specify a building name in your query."
    
    # Create the map
    rag = RealEstateRAG(llm, embedding_client)
    map_obj = rag.create_commute_map(building_name, amenity_types, mode)
    
    if not map_obj:
        return f"Could not create a map for building: {building_name}"
    
    # Save the map to a temporary HTML file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        map_obj.save(tmp.name)
        return f"Map created for {building_name}. Use st.components.html to display it."

# Add this to your Streamlit app
def display_commute_map(building_name, amenity_types=None, mode="driving"):
    """
    Display a commute map in Streamlit
    """
    st.subheader(f"Commute Map for {building_name}")
    
    # Create travel mode selector
    travel_mode = st.radio(
        "Travel Mode",
        ["driving", "walking", "transit"],
        horizontal=True,
        index=["driving", "walking", "transit"].index(mode)
    )
    
    # Create amenity type selector
    all_amenity_types = [
        "MetroStation", "Transport", "Supermarket", "Restaurant", 
        "School", "Hospital", "Park", "Mall", "Gym", "SPA", 
        "Worship", "TouristAttraction"
    ]
    
    selected_amenities = st.multiselect(
        "Select Amenity Types",
        all_amenity_types,
        default=amenity_types if amenity_types else ["MetroStation", "Supermarket", "Restaurant"]
    )
    
    # Create the map
    rag = RealEstateRAG(llm, embedding_client)
    map_obj = rag.create_commute_map(building_name, selected_amenities, travel_mode)
    
    if map_obj:
        # Display the map
        folium_static(map_obj)
    else:
        st.error(f"Could not create map for building: {building_name}")

# Set page configuration
st.set_page_config(
    page_title="Dubai Property Advisor",
    page_icon="🏢",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header and description
st.title("🏢 GAC Dubai Property AI")
st.markdown("""
By team GAC
""")



def streamlit_app():
    
    query = st.text_area("Enter your query:", "I'm looking for a 2-bedroom apartment near good schools with a short commute to Downtown Dubai")
    
    if st.button("Plan my move"):
        with st.spinner("Analyzing your request..."):
            answer = get_property_recommendations(query)
            
            st.markdown(answer)
            
def process_response(response):
    """Process the response from the crew and display any maps"""
    # Check if the response contains a map visualization request
    import re
    map_pattern = r'\[MAP_VISUALIZATION: (.*?)\]'
    map_matches = re.findall(map_pattern, response)
    
    # Display the text response without the map tags
    clean_response = re.sub(map_pattern, '', response)
    st.markdown(clean_response)
    
    # Process any map visualization requests
    for map_request in map_matches:
        parts = [p.strip() for p in map_request.split(',')]
        if len(parts) > 0:
            building_name = parts[0]
            amenity_types = parts[1:] if len(parts) > 1 else None
            display_commute_map(building_name, amenity_types)

st.title("Dubai Real Estate Assistant")

query = st.text_input("")
if query:
    with st.spinner("Analyzing your query..."):
        response = get_property_recommendations(query)
    
    process_response(response)