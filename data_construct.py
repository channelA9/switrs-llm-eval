import pandas as pd
import numpy as np

# Constants
DEFAULT_DISTANCE_THRESHOLD = 400  # Distance threshold in meters, eg 100m, 500m, 5km
MAX_ENTRIES = 100
# Load datasets
traffic = pd.read_csv("datasets/switrs.csv", delimiter=",", skipinitialspace=True)
streetstory = pd.read_csv("datasets/streetstory.csv", on_bad_lines="skip", sep=";")

traffic["latitude"] = pd.to_numeric(traffic["latitude"], errors="coerce")
traffic["longitude"] = pd.to_numeric(traffic["longitude"], errors="coerce")


# Haversine formula to compute distance (in meters) between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # Distance in meters


# Function to find entries within a specified radius
def find_entries_within_radius(
    latitude,
    longitude,
    dataset,
    radius=DEFAULT_DISTANCE_THRESHOLD,
    max_entries=MAX_ENTRIES,
):
    # Compute distances
    dataset["distance"] = dataset.apply(
        lambda row: haversine(latitude, longitude, row["latitude"], row["longitude"]),
        axis=1,
    )
    # Filter entries within the specified radius
    nearby_entries = dataset[dataset["distance"] <= radius]
    # If the number of entries exceeds max_entries, select the closest ones
    if len(nearby_entries) > max_entries:
        nearby_entries = nearby_entries.nsmallest(max_entries, "distance")
    return nearby_entries


# Function to format entries into a text string
def format_entries(entries, dataset_name):
    entry_strings = []
    if entries.empty:
        return f"No {dataset_name} entries found within the specified radius.\n"
    elif dataset_name == "SWITRS":
        # Index,accident_year,latitude,longitude,primary_rd,secondary_rd,population,number_killed,alcohol_involved,stwd_vehtype_at_fault,case_id,number_injured,distance,direction,month,day,year,day_of_week,hour,minute,city_code,city_name,county_code,pcf_code,collision_type_code,mviw_code,lighting_code,severity_code,road_surface_code,weather_code,dvmt

        for _, row in entries.iterrows():
            # - **Fatalities:** {row['number_killed']}
            # - **Injuries:** {row['number_injured']}
            # - **Coordinates:** ({row['latitude']}, {row['longitude']})

            entry_info = f"""
Incident {row['case_id']}
- **Year:** {row['accident_year']}
- **Primary Road:** {row['primary_rd']}
- **Secondary Road:** {row['secondary_rd']}
- **Number of people involved:** {row['population']}
- **Fatalities:** {row['number_killed']}
- **Injuries:** {row['number_injured']}
- **Alcohol Involved:** {row['alcohol_involved']} (1 = Yes, 0 = No)
- **Vehicle at Fault:** {row['stwd_vehtype_at_fault']}
- **Collision Type Code:** {row['collision_type_code']}
- **Lighting Conditions:** {row['lighting_code']}
- **Road Surface Conditions:** {row['road_surface_code']}
- **Weather Conditions:** {row['weather_code']}
- **Time of Accident:** {row['hour']}:{row['minute']} on {row['day_of_week']}, {row['month']}/{row['day']}/{row['year']}
- **Distance from Query Point:** {row['distance']} meters
- **City:** {row['city_name']}, **County:** {row['county_code']}"""
            entry_strings.append(entry_info)
    elif dataset_name == "StreetStory":
        # OBJECTID;type;crash/near-miss date;crash/near-miss mode;crash/near-miss time of day;mode involved;e-scooter involved;Was anyone injured?;crash cause;crash narrative;nearmiss cause;nearmiss narrative;police;unsafe location mode;unsafe location time of day;unsafe location cause;unsafe location narrative;safe location mode;safe location cause;safe location narrative;mobility device;e-bike;e-scooter;improvement;improvement narrative;address;report date;latitude;longitude;type;coordinates

        for _, row in entries.iterrows():
            # **Coordinates:** ({latitude}, {longitude})
            entry_info = f"""
Community Report {row['OBJECTID']}
- **Incident Type:** {row['type']}
- **Date:** {row['crash/near-miss date']}
- **Mode Involved:** {row['crash/near-miss mode']} (e.g., Pedestrian, Bicycle, Car)
- **Time of Incident:** {row['crash/near-miss time of day']}
- **Injuries Reported:** {row['Was anyone injured?']}
- **Cause of Incident:** {row['crash cause']} | **Narrative:** {row['crash narrative']}
- **Near-Miss Cause:** {row['nearmiss cause']} | **Narrative:** {row['nearmiss narrative']}
- **Unsafe Location Details:** {row['unsafe location narrative']}
- **Suggested Infrastructure Improvements:** {row['improvement narrative']}
- **Address:** {row['address']}
- **Report Date:** {row['report date']}
- **Distance from Query Point:** {row['distance']} meters"""
            entry_strings.append(entry_info)

    return "\n".join(entry_strings) + "\n"


# Main function to generate the prompt for the LLM
def generate_location_danger_prompt(latitude, longitude, distance_threshold=DEFAULT_DISTANCE_THRESHOLD):
    traffic_entries = find_entries_within_radius(latitude, longitude, traffic, distance_threshold)
    streetstory_entries = find_entries_within_radius(latitude, longitude, streetstory, distance_threshold)

    prompt = f"""
    You are an AI model tasked with assessing crash risk at a given location based on historical traffic collision and near-miss reports. Given the following structured data, analyze patterns and generate an assessment of potential risks in the specified area strictly adhering to the output guidelines.

    ## **Location of Interest**
    - **Latitude:** {latitude}
    - **Longitude:** {longitude}
    - **Radius of Analysis:** {distance_threshold} meters

    ## **Traffic Collision Data (Past Incidents)**
    {format_entries(traffic_entries, 'SWITRS')}

    ## **StreetStory Reports (Near-Miss and Safety Reports)**
    {format_entries(streetstory_entries, 'StreetStory')}

    ## **Analysis Task**
    - Identify recurring patterns in collisions and near-miss reports within the 500m radius.
    - Determine potential risk factors based on past incidents (e.g., lighting conditions, alcohol involvement, road surface issues, weather, time of day).
    - Assess whether specific infrastructure improvements have been suggested by past reports.
    - Provide an overall risk assessment for this location, considering all available data.

    ## **Output Format**
    Predicted Collisions in 2024: [INTEGER]
    Predicted Fatalities in 2024: [INTEGER]
    Reasoning: [A short, non-technical justification of your prediction in plaintext]
    Recommendations: [Plaintext recommendations, if any, based on the analysis and potential trends visible in the given data]]
    """
    return prompt