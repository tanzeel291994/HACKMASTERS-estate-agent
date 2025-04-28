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

def display_property_map(building_names, amenity_types=None):
    """
    Display a map with buildings, amenities, and commute routes in Streamlit
    
    Parameters:
    - building_names: List of building names to display
    - amenity_types: List of amenity types to include
    """
    if not building_names:
        st.warning("No buildings found to display on the map.")
        return
    
    rag = RealEstateRAG(llm, embedding_client)
    
    # Create tabs for each building
    tabs = st.tabs(building_names)
    
    for i, tab in enumerate(tabs):
        with tab:
            building_name = building_names[i]
            
            # Allow user to select amenity types
            if not amenity_types:
                amenity_options = [
                    "School", "Hospital", "MetroStation", "Transport", 
                    "Supermarket", "Mall", "Restaurant", "Park", 
                    "Gym", "SPA", "Worship", "TouristAttraction"
                ]
                selected_amenities = st.multiselect(
                    "Select amenities to display:",
                    amenity_options,
                    default=["School", "Hospital", "MetroStation", "Supermarket"]
                )
            else:
                selected_amenities = amenity_types

             # Option to show routes
            show_routes = st.checkbox("Show commute routes", value=True)
            
            # Create the map
            map_obj = rag.create_commute_map(building_name, selected_amenities, show_routes)
            
            if map_obj:
                # Display the map
                folium_static(map_obj)
                
                # Display commute times if routes are shown
                if show_routes:
                    st.subheader("Estimated Commute Times")
                    
                    amenities = rag.get_nearby_amenities(building_name, selected_amenities)
                    building = rag.get_building_coordinates(building_name)
                    
                    if building and amenities:
                        # Create a table of commute times
                        commute_data = []
                        
                        origin_lat = float(building["latitude"])
                        origin_lng = float(building["longitude"])
                        
                        # Get morning and evening peak hours
                        tomorrow = datetime.now() + timedelta(days=1)
                        morning_peak = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 8, 0, 0)
                        morning_timestamp = int(morning_peak.timestamp())
                        
                        evening_peak = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 18, 0, 0)
                        evening_timestamp = int(evening_peak.timestamp())

                        for amenity in amenities:
                            dest_lat = float(amenity["latitude"])
                            dest_lng = float(amenity["longitude"])
                            
                            # Get directions for morning commute
                            morning_route = get_directions(
                                origin_lat, origin_lng, 
                                dest_lat, dest_lng,
                                departure_time=morning_timestamp
                            )
                            
                            # Get directions for evening commute
                            evening_route = get_directions(
                                dest_lat, dest_lng,
                                origin_lat, origin_lng,
                                departure_time=evening_timestamp
                            )
                            
                            if morning_route and evening_route:
                                commute_data.append({
                                    "Amenity": amenity["name"],
                                    "Type": amenity["type"],
                                    "Morning Commute": morning_route["duration_in_traffic"],
                                    "Evening Commute": evening_route["duration_in_traffic"],
                                    "Distance": morning_route["distance"]
                                })
                        
                        if commute_data:
                            st.table(commute_data)
            else:
                st.error(f"Could not create map for {building_name}")

def streamlit_app():
    
    query = st.text_area("Enter your query:", "I'm looking for a 2-bedroom apartment near good schools with a short commute to Downtown Dubai")
    
    if st.button("Plan my move"):
        with st.spinner("Analyzing your request..."):
            answer = get_property_recommendations(query)
            
            st.markdown(answer)
            

if __name__ == "__main__" and "streamlit" in sys.modules:
    streamlit_app()
else:
    # Test the agent
    query = "I'm looking for a 2-bedroom apartment with good rated schools"
    recommendation = get_property_recommendations(query)
    print(recommendation)