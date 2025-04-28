import requests
import os

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Function to get directions and travel time between two points
def get_directions(origin_lat, origin_lng, dest_lat, dest_lng, mode="driving", departure_time=None):
    """
    Get directions and travel time between two points using Google Maps Directions API
    
    Parameters:
    - origin_lat, origin_lng: Coordinates of starting point
    - dest_lat, dest_lng: Coordinates of destination
    - mode: Travel mode (driving, walking, bicycling, transit)
    - departure_time: Departure time as Unix timestamp (for traffic estimation)
    
    Returns:
    - Dictionary with route information, polyline, and duration
    """
    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    
    params = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": f"{dest_lat},{dest_lng}",
        "mode": mode,
        "key": GOOGLE_MAPS_API_KEY
    }
    
    # Add departure_time for traffic estimation if provided
    if departure_time:
        params["departure_time"] = departure_time
        params["traffic_model"] = "best_guess"  # Options: best_guess, pessimistic, optimistic
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if data["status"] != "OK":
        print(f"Error getting directions: {data['status']}")
        return None
    
    route = data["routes"][0]
    leg = route["legs"][0]
    
    # Extract polyline for the route
    route_polyline = route["overview_polyline"]["points"]
    decoded_polyline = polyline.decode(route_polyline)
    
    # Extract duration information
    duration_in_traffic = leg.get("duration_in_traffic", {}).get("text", "N/A")
    normal_duration = leg["duration"]["text"]
    
    return {
        "polyline": decoded_polyline,
        "normal_duration": normal_duration,
        "duration_in_traffic": duration_in_traffic,
        "distance": leg["distance"]["text"],
        "start_address": leg["start_address"],
        "end_address": leg["end_address"],
        "steps": leg["steps"]
    }

# Function to create a map with buildings, amenities, and routes
def create_map(building_data, amenities_data=None, show_route=False):
    """
    Create a Folium map with buildings, amenities, and optional routes
    
    Parameters:
    - building_data: List of buildings with coordinates
    - amenities_data: List of amenities with coordinates
    - show_route: Boolean to indicate if routes should be shown
    
    Returns:
    - Folium map object
    """
    # Create a map centered on the first building
    if building_data and len(building_data) > 0:
        center_lat = float(building_data[0].get("latitude", 25.2048))
        center_lng = float(building_data[0].get("longitude", 55.2708))
    else:
        # Default to Dubai center if no buildings
        center_lat, center_lng = 25.2048, 55.2708
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13)
    
    # Add buildings to the map
    for building in building_data:
        if "latitude" in building and "longitude" in building:
            lat = float(building["latitude"])
            lng = float(building["longitude"])
            name = building.get("name", "Unnamed Building")
            
            popup_html = f"""
            <b>{name}</b><br>
            {building.get('description', '')}
            """
            
            folium.Marker(
                location=[lat, lng],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color="blue", icon="home")
            ).add_to(m)
    
    # Add amenities to the map if provided
    if amenities_data:
        for amenity in amenities_data:
            if "latitude" in amenity and "longitude" in amenity:
                lat = float(amenity["latitude"])
                lng = float(amenity["longitude"])
                name = amenity.get("name", "Unnamed Amenity")
                amenity_type = amenity.get("type", "Unknown")
                
                # Choose icon based on amenity type
                icon_map = {
                    "School": "graduation-cap",
                    "Hospital": "plus",
                    "MetroStation": "subway",
                    "Transport": "bus",
                    "Supermarket": "shopping-cart",
                    "Mall": "shopping-bag",
                    "Restaurant": "cutlery",
                    "Park": "tree",
                    "Gym": "heartbeat",
                    "SPA": "spa",
                    "Worship": "building",
                    "TouristAttraction": "camera"
                }
                
                icon_color_map = {
                    "School": "green",
                    "Hospital": "red",
                    "MetroStation": "purple",
                    "Transport": "orange",
                    "Supermarket": "darkblue",
                    "Mall": "pink",
                    "Restaurant": "cadetblue",
                    "Park": "darkgreen",
                    "Gym": "darkpurple",
                    "SPA": "lightred",
                    "Worship": "gray",
                    "TouristAttraction": "beige"
                }
                
                icon_name = icon_map.get(amenity_type, "info-sign")
                icon_color = icon_color_map.get(amenity_type, "gray")
                
                popup_html = f"""
                <b>{name}</b><br>
                Type: {amenity_type}<br>
                Rating: {amenity.get('rating', 'N/A')}<br>
                {amenity.get('vicinity', '')}
                """
                
                folium.Marker(
                    location=[lat, lng],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
                ).add_to(m)
    
    # Add routes if requested
    if show_route and building_data and amenities_data:
        # Get the first building as origin
        origin_lat = float(building_data[0]["latitude"])
        origin_lng = float(building_data[0]["longitude"])
        
        # For each amenity, show a route
        for amenity in amenities_data:
            if "latitude" in amenity and "longitude" in amenity:
                dest_lat = float(amenity["latitude"])
                dest_lng = float(amenity["longitude"])
                
                # Get morning peak hour (8 AM next day)
                tomorrow = datetime.now() + timedelta(days=1)
                morning_peak = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 8, 0, 0)
                morning_timestamp = int(morning_peak.timestamp())
                
                # Get evening peak hour (6 PM next day)
                evening_peak = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 18, 0, 0)
                evening_timestamp = int(evening_peak.timestamp())
                
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
                
                if morning_route:
                    # Add the route line to the map
                    folium.PolyLine(
                        morning_route["polyline"],
                        color="blue",
                        weight=3,
                        opacity=0.7,
                        popup=f"Morning: {morning_route['duration_in_traffic']}"
                    ).add_to(m)
                
                if evening_route:
                    # Add the route line to the map
                    folium.PolyLine(
                        evening_route["polyline"],
                        color="red",
                        weight=3,
                        opacity=0.7,
                        popup=f"Evening: {evening_route['duration_in_traffic']}"
                    ).add_to(m)
    
    return m