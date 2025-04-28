import streamlit as st
import folium
from streamlit_folium import folium_static
import requests
import polyline
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
# Set your Google Maps API key
# For a hackathon, you can hardcode it here or use st.secrets in Streamlit
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAP_KEY")

def get_directions(origin_lat, origin_lng, dest_lat, dest_lng):
    """Simple function to get directions between two points"""
    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # Get tomorrow morning 8 AM for traffic estimation
    tomorrow = datetime.now() + timedelta(days=1)
    morning_peak = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 8, 0, 0)
    morning_timestamp = int(morning_peak.timestamp())
    
    params = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": f"{dest_lat},{dest_lng}",
        "mode": "driving",
        "departure_time": morning_timestamp,
        "traffic_model": "best_guess",
        "key": GOOGLE_MAPS_API_KEY
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if data["status"] != "OK":
        st.error(f"Error getting directions: {data['status']}")
        return None
    
    route = data["routes"][0]
    leg = route["legs"][0]
    
    # Extract polyline for the route
    route_polyline = route["overview_polyline"]["points"]
    decoded_polyline = polyline.decode(route_polyline)
    
    # Extract duration information
    duration_in_traffic = leg.get("duration_in_traffic", {}).get("text", "N/A")
    
    return {
        "polyline": decoded_polyline,
        "duration": duration_in_traffic,
        "distance": leg["distance"]["text"]
    }

def show_property_school_map():
    st.title("Dubai Property School Commute")
    st.write("See the commute time between a property and a nearby school")
    
    # Sample data - replace with your actual data from Neo4j
    # For hackathon purposes, we'll use hardcoded values
    building = {
        "name": "Marina Arcade",
        "latitude": 25.080406,
        "longitude": 55.142700,
        "description": "Luxury apartment building in Dubai Marina"
    }
    
    school = {
        "name": "Dubai American Academy",
        "latitude": 25.113855,
        "longitude": 55.165693,
        "type": "School",
        "rating": 4.5
    }
    
    # Create a map centered on the building
    m = folium.Map(location=[building["latitude"], building["longitude"]], zoom_start=13)
    
    # Add building marker
    folium.Marker(
        location=[building["latitude"], building["longitude"]],
        popup=folium.Popup(f"<b>{building['name']}</b><br>{building['description']}", max_width=300),
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(m)
    
    # Add school marker
    folium.Marker(
        location=[school["latitude"], school["longitude"]],
        popup=folium.Popup(f"<b>{school['name']}</b><br>Rating: {school['rating']}", max_width=300),
        icon=folium.Icon(color="green", icon="graduation-cap", prefix="fa")
    ).add_to(m)
    
    # Get directions
    route_info = get_directions(
        building["latitude"], building["longitude"],
        school["latitude"], school["longitude"]
    )
    
    if route_info:
        # Add the route line to the map
        folium.PolyLine(
            route_info["polyline"],
            color="blue",
            weight=3,
            opacity=0.7,
            popup=f"Morning commute: {route_info['duration']}"
        ).add_to(m)
        
        # Display commute information
        st.subheader(f"Commute: {building['name']} to {school['name']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Distance", route_info["distance"])
        with col2:
            st.metric("Morning Commute", route_info["duration"])
        
        # Display the map
        folium_static(m)
        
        # Additional information
        st.info("""
        - Blue line shows the morning commute route (8:00 AM)
        - Commute time includes traffic conditions
        - Data is based on Google Maps traffic predictions
        """)
    else:
        st.error("Could not calculate route information.")

# Run the app
if __name__ == "__main__":
    show_property_school_map()