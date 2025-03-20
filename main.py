from langchain_google_genai import ChatGoogleGenerativeAI

from data_construct import generate_location_danger_prompt

DISTANCE_THRESHOLD = 200 

latitude, longitude = 34.011845872718396, -117.43391615200592
latitude = round(latitude, 6)
longitude = round(longitude, 6)

llm_prompt = generate_location_danger_prompt(latitude, longitude, DISTANCE_THRESHOLD)

chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key='')
response = chat.invoke(llm_prompt)

print(response.content)