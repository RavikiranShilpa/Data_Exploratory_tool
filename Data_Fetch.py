#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to fetch agriculture data from the USDA API
def fetch_usda_agriculture_data(api_key, crop_name):
    base_url = 'https://api.nal.usda.gov/fdc/v1/foods/search'

    params = {
        'query': crop_name,
        'api_key': api_key
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        api_data = response.json()

        # Assuming the first result in the search is the crop we want
        if api_data['foods']:
            food_id = api_data['foods'][0]['fdcId']

            # Fetch details for the specific food item
            details_url = f'https://api.nal.usda.gov/fdc/v1/food/{food_id}'
            details_params = {'api_key': api_key}

            details_response = requests.get(details_url, params=details_params)
            if details_response.status_code == 200:
                details_data = details_response.json()

                # Extract relevant information
                return {
                    'Crop': crop_name,
                    'Planting_Date': details_data.get('foodNutrients', {}).get('foodNutrientDerivation', ''),
                    'Harvest_Date': details_data.get('foodNutrients', {}).get('yieldAmount', 0),
                    'Fertilizer_Kg': details_data.get('foodNutrients', {}).get('fertilizerUsage', 0),
                    'Water_Usage_mm': details_data.get('foodNutrients', {}).get('waterUsage', 0)
                }
            else:
                print(f"Error fetching details for {crop_name}. Status code: {details_response.status_code}")
    else:
        print(f"Error searching for {crop_name}. Status code: {response.status_code}")

    return None

# Function to fetch weather data from the OpenWeatherMap API
def fetch_weather_data(api_key):
    city = 'YourCity'  # Replace with the name of your city
    api_url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'

    response = requests.get(api_url)
    if response.status_code == 200:
        api_data = response.json()
        return {
            'Date': datetime.utcnow().strftime('%Y-%m-%d'),
            'Temperature_C': api_data['main']['temp'],
            'Humidity_Percent': api_data['main']['humidity'],
            'Wind_Speed_Kmph': api_data['wind']['speed']
        }
    else:
        print(f"Error fetching weather data from API. Status code: {response.status_code}")
        return None

# Function to fetch economic data from the World Bank API
def fetch_economic_data():
    indicator_code = 'NY.GDP.MKTP.CD'  # Replace with the World Bank indicator code
    api_url = f'http://api.worldbank.org/v2/country/USA/indicator/{indicator_code}?format=json'

    response = requests.get(api_url)
    if response.status_code == 200:
        api_data = response.json()
        if api_data[1]:
            return {'Year': datetime.utcnow().year, 'GDP_Billion_USD': api_data[1][0]['value']}
        else:
            print("No economic data available for the specified indicator.")
            return None
    else:
        print(f"Error fetching economic data from API. Status code: {response.status_code}")
        return None
# Fetch agriculture data
agriculture_data = []
crops = ['Wheat', 'Corn', 'Rice', 'Barley', 'Soybean']
for crop in crops:
    usda_data = fetch_usda_agriculture_data(usda_api_key, crop)
    if usda_data:
        agriculture_data.append(usda_data)

# Fetch weather data
weather_data = [fetch_weather_data(openweathermap_api_key)]

# Fetch economic data
economic_data = [fetch_economic_data()]

# Create DataFrames from fetched data
agriculture_df = pd.DataFrame(agriculture_data)
weather_df = pd.DataFrame(weather_data)
economy_df = pd.DataFrame(economic_data)
#agriculture_df=pd.read_csv("agriculture_df")
#weather_df=pd.read_csv("weather_df")
#economy_df=pd.read_csv("economy_df")
# Merge datasets
merged_df = pd.merge(agriculture_df, weather_df, how='inner', on='Date')
merged_df = pd.merge(merged_df, economy_df, how='inner', on='Year')

