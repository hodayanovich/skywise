# skywise.py - Professional Weather Intelligence Platform
# A complete Dash web application for weather forecasting with ML predictions

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, no_update
import dash_bootstrap_components as dbc
import requests
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ], 
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True
)

# =============================================================================
# CSS STYLING AND THEME
# =============================================================================

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>SkyWise - Professional Weather Intelligence</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-blue: #1e40af;
                --light-blue: #3b82f6;
                --accent-blue: #60a5fa;
                --accent-yellow: #f59e0b;
                --light-yellow: #fef3c7;
                --dark-yellow: #d97706;
                --gray-50: #f8fafc;
                --gray-100: #f1f5f9;
                --gray-600: #475569;
                --gray-800: #1e293b;
                --white: #ffffff;
            }
            
            body {
                background-color: var(--gray-50);
                color: var(--gray-800);
            }
            
            .suggestion-item {
                padding: 12px 16px;
                border-bottom: 1px solid #e2e8f0;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
            }
            
            .suggestion-item:hover {
                background-color: var(--light-yellow) !important;
                color: var(--primary-blue) !important;
                transform: translateX(4px);
            }
            
            .suggestion-item:last-child {
                border-bottom: none;
            }
            
            .weather-card {
                background: var(--white);
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                box-shadow: 0 4px 16px rgba(30, 64, 175, 0.08);
                transition: all 0.3s ease;
            }
            
            .weather-card:hover {
                box-shadow: 0 8px 24px rgba(30, 64, 175, 0.12);
            }
            
            .professional-header {
                background: linear-gradient(135deg, var(--primary-blue) 0%, var(--light-blue) 100%);
                box-shadow: 0 4px 16px rgba(30, 64, 175, 0.15);
            }
            
            .temperature-display {
                background: linear-gradient(135deg, var(--primary-blue), var(--light-blue));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: 700;
            }
            
            .beta-card {
                background: linear-gradient(135deg, var(--accent-yellow), var(--dark-yellow));
                border: none;
                border-radius: 12px;
                color: var(--white);
                transition: all 0.3s ease;
                height: 100%;
            }
            
            .beta-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(245, 158, 11, 0.3);
            }
            
            .ml-prediction-card {
                background: linear-gradient(135deg, #10b981, #059669);
                border: none;
                border-radius: 12px;
                color: var(--white);
                transition: all 0.3s ease;
            }
            
            .ml-prediction-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3);
            }
            
            .location-badge {
                background: linear-gradient(45deg, var(--light-blue), var(--primary-blue));
                border: none;
                color: var(--white);
                border-radius: 20px;
                font-size: 0.8rem;
                padding: 0.4rem 0.8rem;
            }
            
            .search-container {
                position: relative;
                max-width: 500px;
                margin: 0 auto;
            }
            
            .search-input {
                border: 2px solid #e2e8f0;
                background: var(--white);
                font-size: 16px;
                padding: 14px 20px;
                border-radius: 50px 0 0 50px;
                box-shadow: 0 2px 8px rgba(30, 64, 175, 0.06);
                transition: all 0.3s ease;
            }
            
            .search-input:focus {
                border-color: var(--light-blue);
                box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
                outline: none;
            }
            
            .search-btn {
                background: var(--primary-blue);
                border: 2px solid var(--primary-blue);
                color: var(--white);
                border-radius: 0 50px 50px 0;
                width: 50px;
                transition: all 0.3s ease;
            }
            
            .search-btn:hover {
                background: var(--light-blue);
                border-color: var(--light-blue);
                transform: translateY(-1px);
            }
            
            .suggestions-dropdown {
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background: var(--white);
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(30, 64, 175, 0.12);
                z-index: 1000;
                max-height: 300px;
                overflow-y: auto;
                margin-top: 4px;
            }
            
            .suggestions-header {
                padding: 8px 16px;
                background-color: var(--gray-50);
                font-weight: 600;
                font-size: 0.85rem;
                color: var(--gray-600);
                border-bottom: 1px solid #e2e8f0;
            }
            
            .text-primary { color: var(--primary-blue) !important; }
            .text-warning { color: var(--accent-yellow) !important; }
            .badge-warning { background-color: var(--accent-yellow) !important; color: var(--white) !important; }
            .badge-primary { background-color: var(--primary-blue) !important; }
            
            /* Improved welcome message styling */
            .welcome-city-badge {
                background-color: var(--primary-blue) !important;
                color: var(--white) !important;
                border: 2px solid var(--primary-blue);
                font-weight: 600;
                padding: 8px 16px;
                font-size: 0.9rem;
                transition: all 0.2s ease;
            }
            
            .welcome-city-badge:hover {
                background-color: var(--white) !important;
                color: var(--primary-blue) !important;
                border-color: var(--primary-blue);
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(30, 64, 175, 0.2);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

NOMINATIM_URL: str = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_URL: str = "https://api.open-meteo.com/v1/forecast"

POPULAR_CITIES: List[str] = [
    "London", "New York", "Tokyo", "Paris", "Sydney", "Berlin", 
    "Los Angeles", "Chicago", "Toronto", "Amsterdam", "Barcelona", "Tel Aviv"
]

WEATHER_CODES: Dict[int, Dict[str, str]] = {
    0: {"description": "Clear sky", "icon": "01d"},
    1: {"description": "Mainly clear", "icon": "01d"},
    2: {"description": "Partly cloudy", "icon": "02d"},
    3: {"description": "Overcast", "icon": "03d"},
    45: {"description": "Fog", "icon": "50d"},
    48: {"description": "Depositing rime fog", "icon": "50d"},
    51: {"description": "Light drizzle", "icon": "09d"},
    53: {"description": "Moderate drizzle", "icon": "09d"},
    55: {"description": "Dense drizzle", "icon": "09d"},
    61: {"description": "Slight rain", "icon": "10d"},
    63: {"description": "Moderate rain", "icon": "10d"},
    65: {"description": "Heavy rain", "icon": "10d"},
    71: {"description": "Slight snow fall", "icon": "13d"},
    73: {"description": "Moderate snow fall", "icon": "13d"},
    75: {"description": "Heavy snow fall", "icon": "13d"},
    80: {"description": "Slight rain showers", "icon": "09d"},
    81: {"description": "Moderate rain showers", "icon": "09d"},
    82: {"description": "Violent rain showers", "icon": "09d"},
    95: {"description": "Thunderstorm", "icon": "11d"},
    96: {"description": "Thunderstorm with slight hail", "icon": "11d"},
    99: {"description": "Thunderstorm with heavy hail", "icon": "11d"}
}

# =============================================================================
# API FUNCTIONS
# =============================================================================

# Geocoding with comprehensive error handling and address parsing
def get_detailed_location(city: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Get detailed location information using OpenStreetMap Nominatim API.
    
    Args:
        city (str): City name to search for
        
    Returns:
        Tuple[Optional[Dict], Optional[str]]: (location_info dict, error_message str)
            One will be None. If successful, location_info contains lat, lon, display_name,
            full_address, and details dict with city, state, country, postcode.
    """
    try:
        url = f"{NOMINATIM_URL}?q={city}&format=json&limit=5&addressdetails=1"
        headers = {'User-Agent': 'SkyWise/1.0 (Python/Dash)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                location = data[0]
                address = location.get('address', {})
                
                city_name = (address.get('city') or address.get('town') or 
                           address.get('village') or address.get('municipality') or
                           location.get('display_name', '').split(',')[0])
                
                state = address.get('state') or address.get('region')
                country = address.get('country')
                postcode = address.get('postcode')
                
                location_info = {
                    'lat': float(location['lat']),
                    'lon': float(location['lon']),
                    'display_name': city_name,
                    'full_address': location.get('display_name', ''),
                    'details': {
                        'city': city_name,
                        'state': state,
                        'country': country,
                        'postcode': postcode
                    }
                }
                
                return location_info, None
            else:
                return None, f"Location '{city}' not found."
        else:
            return None, f"Geocoding error: {response.status_code}"
    except Exception as e:
        return None, f"Error: {str(e)}"


# Main weather data fetcher with past_days=15 for ML training data
def get_weather_data(city: str, units: str = "metric") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetch comprehensive weather data using Open-Meteo API.
    
    Args:
        city (str): City name to get weather for
        units (str): 'metric' for Celsius or 'imperial' for Fahrenheit
        
    Returns:
        Tuple[Optional[Dict], Optional[str]]: (weather_data dict, error_message str)
            One will be None. Weather data contains current, hourly, and daily forecasts.
    """
    try:
        location_info, error = get_detailed_location(city)
        if error:
            return None, error
        
        lat, lon = location_info['lat'], location_info['lon']
        temp_unit = "celsius" if units == "metric" else "fahrenheit"
        
        url = (f"{OPEN_METEO_URL}?"
               f"latitude={lat}&longitude={lon}&"
               f"current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation&"
               f"hourly=temperature_2m,weather_code,precipitation_probability&"
               f"daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max&"
               f"temperature_unit={temp_unit}&"
               f"past_days=15&"
               f"wind_speed_unit=ms&"
               f"timezone=auto&"
               f"forecast_days=5")
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            current_weather_code = data['current']['weather_code']
            weather_info = WEATHER_CODES.get(current_weather_code, {"description": "Unknown", "icon": "01d"})
            
            weather_data = {
                'city_name': location_info['display_name'],
                'location_info': location_info,
                'current': {
                    'dt': datetime.fromisoformat(data['current']['time'].replace('T', ' ')).timestamp(),
                    'temp': data['current']['temperature_2m'],
                    'feels_like': data['current']['temperature_2m'],
                    'humidity': data['current']['relative_humidity_2m'],
                    'pressure': 1013,
                    'wind_speed': data['current']['wind_speed_10m'],
                    'precipitation': data['current'].get('precipitation', 0),
                    'weather': [{'description': weather_info['description'], 'icon': weather_info['icon']}]
                },
                'hourly': [],
                'daily': []
            }
            
            # Process hourly data (next 24 hours)
            for i in range(min(24, len(data['hourly']['time']))):
                weather_data['hourly'].append({
                    'dt': datetime.fromisoformat(data['hourly']['time'][i].replace('T', ' ')).timestamp(),
                    'temp': data['hourly']['temperature_2m'][i],
                    'precipitation_probability': data['hourly']['precipitation_probability'][i]
                })
            
            # Process daily data (next 5 days)
            for i in range(min(5, len(data['daily']['time']))):
                daily_weather_code = data['daily']['weather_code'][i]
                daily_weather_info = WEATHER_CODES.get(daily_weather_code, {"description": "Unknown", "icon": "01d"})
                
                weather_data['daily'].append({
                    'dt': datetime.fromisoformat(data['daily']['time'][i]).timestamp(),
                    'temp': {
                        'max': data['daily']['temperature_2m_max'][i],
                        'min': data['daily']['temperature_2m_min'][i]
                    },
                    'precipitation_probability': data['daily']['precipitation_probability_max'][i],
                    'weather': [{'description': daily_weather_info['description'], 'icon': daily_weather_info['icon']}],
                    'is_past': datetime.fromisoformat(data['daily']['time'][i]).date() < datetime.now().date()
                })
            
            return weather_data, None
            
        else:
            return None, f"Weather API error: {response.status_code}"
    except Exception as e:
        return None, f"Error: {str(e)}"

# =============================================================================
# IMPROVED ML PREDICTION FUNCTIONS
# =============================================================================


def predict_tomorrow_temperature(weather_data: dict, units: str = "metric") -> Tuple[Optional[dict], Optional[str]]:
    """
    ML-based temperature prediction using Linear Regression with feature engineering.
    Uses past 10 days to predict tomorrow's average temperature.
    
    Features:
        - day index (trend)
        - previous day's average temperature (autoregression)
        - temperature range (max - min)
        - seasonal component (sin(2*pi*day_of_year/365.25))
    
    Args:
        weather_data (Dict[str, Any]): Weather data containing 'daily' with past 10 days
        units (str): 'metric' or 'imperial'
        
    Returns:
        Tuple[Optional[dict], Optional[str]]: prediction data or error
    """
    try:
        if not weather_data or not weather_data.get('daily'):
            return None, "Insufficient data for prediction"
        
        daily_data = [d for d in weather_data['daily'] if d.get('is_past')]  # Only past days
        if len(daily_data) < 3:
            return None, "Need at least 3 past days for reliable prediction"
        
        #daily_data = daily_data[-10:]  # Use last 10 days max
        X, y = [], []
        prev_avg_temp = None
        
        print("Preparing features from past days:")
        for i, day in enumerate(daily_data):
            avg_temp = (day['temp']['max'] + day['temp']['min']) / 2
            temp_range = day['temp']['max'] - day['temp']['min']
            day_of_year = datetime.fromtimestamp(day['dt']).timetuple().tm_yday
            seasonal = np.sin(2 * np.pi * day_of_year / 365.25)
            
            if prev_avg_temp is None:
                prev_avg_temp = avg_temp  # first day, no previous, just use its own
                
            features = [i, prev_avg_temp, temp_range, seasonal]
            X.append(features)
            y.append(avg_temp)
            
            print(f"Day {i}: avg_temp={avg_temp:.1f}, temp_range={temp_range:.1f}, "
                  f"prev_avg={prev_avg_temp:.1f}, seasonal={seasonal:.3f}")
            
            prev_avg_temp = avg_temp
        
        X = np.array(X)
        y = np.array(y)
        
        print("\nFeature matrix X:")
        print(X)
        print("Target y:")
        print(y)
        
        # Train Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict tomorrow
        next_day_index = len(X)
        tomorrow_day = datetime.fromtimestamp(daily_data[-1]['dt']) + timedelta(days=1)
        tomorrow_day_of_year = tomorrow_day.timetuple().tm_yday
        tomorrow_seasonal = np.sin(2 * np.pi * tomorrow_day_of_year / 365.25)
        avg_temp_range = np.mean([d['temp']['max'] - d['temp']['min'] for d in daily_data[-3:]])
        tomorrow_features = np.array([[next_day_index, y[-1], avg_temp_range, tomorrow_seasonal]])
        
        prediction = model.predict(tomorrow_features)[0]
        
        # Confidence estimate
        y_pred = model.predict(X)
        errors = np.abs(y - y_pred)
        confidence_range = np.mean(errors) + np.std(errors)
        confidence_range = max(1.5, min(5.0, confidence_range))  # reasonable bounds
        
        # Bound prediction reasonably
        temp_min = min(y) - 7
        temp_max = max(y) + 7
        prediction = max(temp_min, min(temp_max, prediction))
        
        unit_symbol = "°C" if units == "metric" else "°F"
        
        print("\nTomorrow features:", tomorrow_features)
        print(f"Predicted avg temp: {prediction:.1f}{unit_symbol} ±{confidence_range:.1f}{unit_symbol}")
        
        return {
            'prediction': round(prediction, 1),
            'confidence_range': round(confidence_range, 1),
            'unit': unit_symbol,
            'model_accuracy': 'Linear Regression with past trend, prev day, temp variability, seasonal component',
            'features_used': ['day_index', 'prev_avg_temp', 'temp_range', 'seasonal_component']
        }, None
    
    except Exception as e:
        return None, f"ML prediction error: {str(e)}"


# =============================================================================
# UI COMPONENT FUNCTIONS
# =============================================================================

def create_navigation_header() -> html.Div:
    """
    Create the professional navigation header with branding and menu links.
    
    Returns:
        html.Div: Navigation header component with SkyWise branding and nav links
    """
    return dbc.Navbar([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.NavbarBrand([
                        html.I(className="fas fa-cloud-sun me-2 text-warning"),
                        "SkyWise"
                    ], className="fs-3 fw-bold text-white")
                ], width="auto"),
                
                dbc.Col([
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink([
                            html.I(className="fas fa-home me-2"),
                            "Weather"
                        ], href="/", className="text-white fw-medium")),
                        dbc.NavItem(dbc.NavLink([
                            html.I(className="fas fa-heart me-2"),
                            "Favorites"
                        ], href="/favorites", className="text-white fw-medium")),
                    ], navbar=True, className="ms-auto")
                ])
            ], className="w-100 align-items-center")
        ], fluid=True)
    ], className="professional-header mb-4", dark=True)

def create_welcome_message() -> dbc.Alert:
    """
    Create the welcome message displayed when no search has been performed.
    
    Returns:
        dbc.Alert: Welcome alert component with popular cities and instructions
    """
    return dbc.Alert([
        html.H4("Welcome to SkyWise", className="alert-heading text-primary"),
        html.P("Professional weather intelligence powered by Open-Meteo", className="fs-5 mb-3"),
        html.P("Search for any city to get detailed weather forecasts with AI insights", className="mb-4"),
        html.Hr(),
        html.H5("Popular Cities", className="mb-3 text-primary"),
        html.Div([
            dbc.Button(city, className="welcome-city-badge me-2 mb-2", 
                     id={'type': 'welcome-city', 'city': city},
                     n_clicks=0)
            for city in POPULAR_CITIES[:8]
        ], className="mb-4"),
        html.P("Or start typing any city name in the search box above", className="text-muted")
    ], color="light", className="text-center border-primary")

def create_error_alert(message: str, suggestion: Optional[str] = None) -> dbc.Alert:
    """
    Create a standardized error alert component with consistent styling.
    
    Args:
        message (str): Main error message to display
        suggestion (Optional[str]): Optional suggestion text for user guidance
        
    Returns:
        dbc.Alert: Error alert component with error message and optional suggestion
    """
    content = [
        html.H5([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Error"
        ]),
        html.P(message)
    ]
    
    if suggestion:
        content.append(html.P(suggestion, className="small text-muted"))
    else:
        content.append(html.P("Please try again with a different search term.", className="small text-muted"))
    
    return dbc.Alert(content, color="danger")

def create_temperature_chart(weather_data: Dict[str, Any], units: str = "metric") -> go.Figure:
    """
    Create an interactive temperature chart using Plotly with hourly forecasts.
    
    Args:
        weather_data (Dict[str, Any]): Weather data containing hourly temperature forecasts
        units (str): Temperature units for display ('metric' or 'imperial')
        
    Returns:
        go.Figure: Plotly figure with temperature trend line and precipitation data
    """
    unit_symbol = "°C" if units == "metric" else "°F"
    
    times = []
    temps = []
    precip_probs = []
    
    for hour in weather_data['hourly']:
        times.append(datetime.fromtimestamp(hour['dt']))
        temps.append(hour['temp'])
        precip_probs.append(hour.get('precipitation_probability', 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times, 
        y=temps,
        mode='lines+markers',
        line=dict(color='#1e40af', width=3),
        marker=dict(size=6, color='#3b82f6'),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)',
        name='Temperature',
        hovertemplate=f'<b>%{{x}}</b><br>Temperature: %{{y}}{unit_symbol}<br>Rain chance: %{{customdata}}%<extra></extra>',
        customdata=precip_probs
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(t=20, b=40, l=40, r=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Time",
        yaxis_title=f"Temperature ({unit_symbol})",
        showlegend=False,
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        font=dict(color='#475569')
    )
    
    return fig

def create_weather_display(weather_data: Dict[str, Any], units: str = "metric", favorites: Optional[List[str]] = None) -> html.Div:
    """
    Create the complete weather display component with current conditions, charts, and forecasts.
    
    Args:
        weather_data (Dict[str, Any]): Complete weather data from API
        units (str): Temperature units ('metric' or 'imperial')  
        favorites (Optional[List[str]]): List of favorite city names
        
    Returns:
        html.Div: Complete weather display with all sections and ML predictions
    """
    current = weather_data['current']
    city = weather_data['city_name']
    location_info = weather_data['location_info']
    unit_symbol = "°C" if units == "metric" else "°F"
    favorites = favorites or []
    
    # Extract current weather details
    temp = round(current['temp'])
    feels_like = round(current['feels_like'])
    description = current['weather'][0]['description'].title()
    humidity = current['humidity']
    wind_speed = round(current['wind_speed'], 1)
    icon = current['weather'][0]['icon']
    
    # Extract today's forecast
    today = weather_data['daily'][0]
    high_temp = round(today['temp']['max'])
    low_temp = round(today['temp']['min'])
    
    # Build location display text
    location_details = location_info['details']
    location_text = city
    if location_details['state']:
        location_text += f", {location_details['state']}"
    if location_details['country']:
        location_text += f", {location_details['country']}"
    if location_details['postcode']:
        location_text += f" {location_details['postcode']}"
    
    # Check if city is in favorites
    is_favorited = city in favorites
    
    # Get improved ML prediction
    ml_prediction, ml_error = predict_tomorrow_temperature(weather_data, units)
    
    return html.Div([
        # Header with location and favorite button
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H3([
                            html.I(className="fas fa-map-marker-alt me-2 text-primary"),
                            city
                        ], className="mb-2"),
                        dbc.Badge(location_text, className="location-badge")
                    ], width=9),
                    dbc.Col([
                        dbc.Button([
                            html.I(className="fas fa-heart me-2" if not is_favorited else "fas fa-heart-broken me-2"),
                            "Remove from Favorites" if is_favorited else "Add to Favorites"
                        ], 
                        id={'type': 'add-favorite-btn', 'city': city},
                        color="danger" if is_favorited else "outline-primary", 
                        size="sm", 
                        className="float-end",
                        n_clicks=0)
                    ], width=3)
                ])
            ])
        ], className="weather-card mb-4"),
        
        # Main weather display row
        dbc.Row([
            # Current weather - left side (8 columns)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H1(f"{temp}{unit_symbol}", className="display-4 temperature-display mb-0"),
                                html.P(f"Feels like {feels_like}{unit_symbol}", className="text-muted mb-2 fs-5"),
                                html.P(description, className="lead mb-3")
                            ], width=8),
                            dbc.Col([
                                html.Img(
                                    src=f"http://openweathermap.org/img/w/{icon}.png",
                                    style={'width': '80px', 'height': '80px'}
                                )
                            ], width=4, className="text-center")
                        ]),
                        
                        # Temperature and precipitation badges
                        html.Div([
                            dbc.Badge([
                                html.I(className="fas fa-arrow-up me-1"),
                                f"High: {high_temp}{unit_symbol}"
                            ], color="danger", className="me-2"),
                            dbc.Badge([
                                html.I(className="fas fa-arrow-down me-1"),
                                f"Low: {low_temp}{unit_symbol}"
                            ], color="info", className="me-2"),
                            dbc.Badge([
                                html.I(className="fas fa-umbrella me-1"),
                                f"Rain: {today['precipitation_probability']}%"
                            ], color="primary")
                        ], className="mb-3"),
                        
                        # Additional weather details
                        dbc.Row([
                            dbc.Col([
                                html.P([
                                    html.I(className="fas fa-tint text-primary me-2"),
                                    f"Humidity: {humidity}%"
                                ], className="mb-1")
                            ], width=6),
                            dbc.Col([
                                html.P([
                                    html.I(className="fas fa-wind text-primary me-2"),
                                    f"Wind: {wind_speed} m/s"
                                ], className="mb-1")
                            ], width=6)
                        ])
                    ])
                ], className="weather-card h-100")
            ], width=8),
            
            # Smart Outfit Beta Feature - right side (4 columns)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.I(className="fas fa-tshirt me-2"),
                            "Smart Outfit",
                            dbc.Badge("BETA", color="warning", className="ms-2")
                        ], className="text-white mb-3 fw-bold"),
                        html.P([
                            html.I(className="fas fa-magic me-2"),
                            "AI-powered clothing recommendations based on current weather conditions and forecast"
                        ], className="mb-3 text-white small"),
                        html.P("Coming Soon!", className="mb-3 text-white fw-bold fs-5"),
                        html.P("Get notified when this feature launches with personalized outfit suggestions.", 
                               className="mb-3 text-white small"),
                        dbc.Button([
                            html.I(className="fas fa-bell me-2"),
                            "Notify Me"
                        ], 
                        id={'type': 'outfit-interest-btn', 'city': city},
                        color="light", 
                        outline=True,
                        size="sm",
                        n_clicks=0)
                    ])
                ], className="beta-card")
            ], width=4)
        ], className="mb-4"),
        
        # Temperature chart section
        dbc.Card([
            dbc.CardBody([
                html.H5([
                    html.I(className="fas fa-chart-line me-2 text-primary"),
                    "24-Hour Temperature Forecast"
                ], className="mb-3"),
                dcc.Graph(
                    figure=create_temperature_chart(weather_data, units),
                    config={'displayModeBar': False}
                )
            ])
        ], className="weather-card mb-4"),
        
        # Enhanced ML Prediction section
        dbc.Card([
            dbc.CardBody([
                html.H5([
                    html.I(className="fas fa-brain me-2"),
                    "Advanced ML Temperature Prediction",
                    dbc.Badge("AI", color="success", className="ms-2")
                ], className="text-white mb-3 fw-bold"),
                
                html.Div([
                    html.H4([
                        html.I(className="fas fa-crystal-ball me-2"),
                        f"Tomorrow's Predicted Average: {ml_prediction['prediction']}{ml_prediction['unit']}" if ml_prediction else "Prediction unavailable"
                    ], className="text-white mb-2") if ml_prediction else html.P(ml_error, className="text-white"),
                    
                    html.P([
                        html.I(className="fas fa-chart-area me-2"),
                        f"Confidence Range: ±{ml_prediction['confidence_range']}{ml_prediction['unit']}" if ml_prediction else ""
                    ], className="text-white small mb-2") if ml_prediction else None,
                    
                    html.P([
                        html.I(className="fas fa-cogs me-2"),
                        ml_prediction['model_accuracy'] if ml_prediction else "Advanced ensemble ML modeling"
                    ], className="text-white small mb-2") if ml_prediction else None,
                    
                    html.P([
                        html.I(className="fas fa-microchip me-2"),
                        f"Models: {ml_prediction['models_used']} | Features: {', '.join(ml_prediction['features'])}" if ml_prediction and 'features' in ml_prediction else ""
                    ], className="text-white small") if ml_prediction and 'features' in ml_prediction else None
                ], className="mb-3")
            ])
        ], className="ml-prediction-card mb-4"),
        
        # 7-day forecast and weekly planner
        dbc.Row([
            # 7-day forecast - left side (8 columns)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="fas fa-calendar-week me-2 text-primary"),
                            "5-Day Forecast"
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.P(datetime.fromtimestamp(day['dt']).strftime('%a'), className="fw-bold mb-1 small"),
                                    html.Img(src=f"http://openweathermap.org/img/w/{day['weather'][0]['icon']}.png", 
                                            style={'width': '35px', 'height': '35px'}),
                                    html.P(f"{round(day['temp']['max'])}{unit_symbol}", className="mb-0 fw-bold small"),
                                    html.P(f"{round(day['temp']['min'])}{unit_symbol}", className="text-muted small mb-1"),
                                    dbc.Badge(f"{day['precipitation_probability']}%", color="info", className="small")
                                ], className="text-center p-2")
                            ], width=12//min(7, len(weather_data['daily'])))
                            for day in weather_data['daily'][:5]
                        ])
                    ])
                ], className="weather-card h-100")
            ], width=8),
            
            # Week Planner Beta Feature - right side (4 columns)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.I(className="fas fa-calendar-check me-2"),
                            "Week Planner",
                            dbc.Badge("BETA", color="primary", className="ms-2")
                        ], className="mb-3 fw-bold"),
                        html.P([
                            html.I(className="fas fa-lightbulb me-2"),
                            "Intelligent activity planning based on weather patterns and forecasts"
                        ], className="mb-3 small"),
                        html.P("Coming Soon!", className="mb-3 fw-bold"),
                        html.P("Plan your week with AI-powered activity suggestions.", className="mb-3 small"),
                        dbc.Button([
                            html.I(className="fas fa-bell me-2"),
                            "Get Updates"
                        ], 
                        id={'type': 'planner-interest-btn', 'city': city},
                        color="primary", 
                        outline=True,
                        size="sm",
                        n_clicks=0)
                    ])
                ], className="beta-card")
            ], width=4)
        ])
    ])

def create_home_page() -> html.Div:
    """
    Create the home page layout with search functionality and temperature unit toggle.
    
    Returns:
        html.Div: Complete home page layout with navigation, search, and results area
    """
    return html.Div([
        create_navigation_header(),
        
        dbc.Container([
            # Temperature unit toggle
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.ButtonGroup([
                            dbc.Button("°C", id="celsius-btn", color="primary", size="sm", n_clicks=0),
                            dbc.Button("°F", id="fahrenheit-btn", color="outline-primary", size="sm", n_clicks=0)
                        ])
                    ], className="text-center mb-4")
                ])
            ]),
            
            # Search section
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.InputGroup([
                            dbc.Input(
                                id='city-input',
                                type='text',
                                placeholder='Search for any city worldwide...',
                                className='search-input',
                                autoComplete="off",
                                n_submit=0
                            ),
                            dbc.Button(
                                html.I(className="fas fa-search"),
                                id='search-btn',
                                n_clicks=0,
                                className='search-btn'
                            )
                        ]),
                        html.Div(id="autofill-suggestions")
                    ], className="search-container")
                ], width=12)
            ], className="mb-5"),
            
            # Weather results section
            dbc.Row([
                dbc.Col([
                    dbc.Spinner(
                        html.Div(
                            create_welcome_message(),
                            id='weather-output'
                        ),
                        color="primary"
                    )
                ])
            ])
        ])
    ])

def create_favorites_page() -> html.Div:
    """
    Create the favorites page layout with header and content area.
    
    Returns:
        html.Div: Complete favorites page layout with navigation and favorites content
    """
    return html.Div([
        create_navigation_header(),
        
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2([
                        html.I(className="fas fa-heart me-2 text-primary"),
                        "Favorite Cities"
                    ], className="text-center mb-4"),
                    html.Div(id='favorites-page-content')
                ])
            ])
        ])
    ])

# =============================================================================
# APP LAYOUT
# =============================================================================

app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='recent-searches', data=[]),
    dcc.Store(id='current-units', data='metric'),
    dcc.Store(id='search-trigger', data=0),
    dcc.Store(id='current-city', data=''),
    dcc.Store(id='favorites-list', data=[]),
    dcc.Store(id='outfit-clicks', data=0),
    dcc.Store(id='planner-clicks', data=0),
    dcc.Store(id='selected-city-from-favorites', data=''),
    html.Div(id='page-content', children=create_home_page())
], fluid=True)

# =============================================================================
# CALLBACK FUNCTIONS 
# =============================================================================

# HOME PAGE CALLBACKS 
@app.callback(
    [Output('weather-output', 'children'),
     Output('current-city', 'data'),
     Output('recent-searches', 'data'),
     Output('selected-city-from-favorites', 'data', allow_duplicate=True),
     Output('city-input', 'value', allow_duplicate=True)],
    [Input('search-btn', 'n_clicks'),
     Input({'type': 'suggestion-click', 'city': ALL}, 'n_clicks'),
     Input({'type': 'welcome-city', 'city': ALL}, 'n_clicks'),
     Input('city-input', 'n_submit'),
     Input('selected-city-from-favorites', 'data')],
    [State('city-input', 'value'),
     State('current-units', 'data'),
     State('recent-searches', 'data'),
     State('favorites-list', 'data'),
     State('url', 'pathname')],
    prevent_initial_call=True
)
def search_weather_home(search_clicks: int, suggestion_clicks: List[int], welcome_clicks: List[int], 
                       enter_submit: int, selected_city_from_favorites: str,
                       city_input_value: str, units: str, recent_searches: List[str], 
                       favorites: List[str], current_path: str) -> Tuple[List, str, List[str], str, str, html.Div]:
    """
    Handle weather searches from the home page and from favorites navigation.
    Fixed to immediately search when suggestions are clicked.
    """
    # Only process searches on home page
    if current_path != "/" and current_path is not None:
        return no_update, no_update, no_update, no_update, no_update
        
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update, no_update

    # Determine what triggered the search and extract city name
    triggered_id = ctx.triggered[0]['prop_id']
    user_action_detected = False
    city_name = city_input_value
    clear_favorites_store = False
    update_input_field = no_update
    
    # Handle city from favorites navigation
    if selected_city_from_favorites:
        user_action_detected = True
        city_name = selected_city_from_favorites
        clear_favorites_store = True
        update_input_field = selected_city_from_favorites
    elif 'search-btn' in triggered_id and search_clicks and search_clicks > 0:
        user_action_detected = True
    elif 'city-input.n_submit' in triggered_id and enter_submit and enter_submit > 0:
        user_action_detected = True
    elif 'suggestion-click' in triggered_id:
        # Fixed: Immediate search on suggestion click
        try:
            button_id = eval(triggered_id.split('.')[0])
            if suggestion_clicks:
                for clicks in suggestion_clicks:
                    if clicks and clicks > 0:
                        user_action_detected = True
                        city_name = button_id['city']
                        update_input_field = city_name  # Update input field
                        break
        except:
            pass
    elif 'welcome-city' in triggered_id:
        # Fixed: Immediate search on welcome city click
        try:
            button_id = eval(triggered_id.split('.')[0])
            if welcome_clicks:
                for clicks in welcome_clicks:
                    if clicks and clicks > 0:
                        user_action_detected = True
                        city_name = button_id['city']
                        update_input_field = city_name  # Update input field
                        break
        except:
            pass

    if not user_action_detected:
        return no_update, no_update, no_update, no_update, no_update

    # Validate city name
    if not city_name or not city_name.strip():
        return ([create_error_alert("Please enter a city name to get weather information")], 
                "", recent_searches or [], "" if clear_favorites_store else no_update, 
                update_input_field)

    # Fetch weather data
    try:
        weather_data, error = get_weather_data(city_name, units or "metric")
        if error:
            return ([create_error_alert(f"Location '{city_name}' not found. Try being more specific (e.g., 'Paris, France' instead of 'Paris')")], 
                    "", recent_searches or [], "" if clear_favorites_store else no_update, 
                    update_input_field)

        # Update recent searches list
        recent_searches = recent_searches or []
        city_title = weather_data['city_name']
        if city_title in recent_searches:
            recent_searches.remove(city_title)
        recent_searches.append(city_title)
        recent_searches = recent_searches[-5:]

        weather_display = create_weather_display(weather_data, units or "metric", favorites)
        return ([weather_display], city_name, recent_searches, 
                "" if clear_favorites_store else no_update, update_input_field)  # Clear suggestions
        
    except Exception as e:
        return ([create_error_alert(f"Error fetching weather data: {str(e)}")], 
                "", recent_searches or [], "" if clear_favorites_store else no_update, 
                update_input_field)

# Handle navigation from favorites to home page
@app.callback(
    [Output('selected-city-from-favorites', 'data'),
     Output('url', 'pathname')],
    [Input({'type': 'view-favorite-btn', 'city': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def navigate_from_favorites(view_favorite_clicks: List[int]) -> Tuple[str, str]:
    """
    Store selected city from favorites and navigate to home page.
    """
    ctx = callback_context
    
    if not ctx.triggered:
        return no_update, no_update
    
    # Check if any actual clicks happened
    if not any(click and click > 0 for click in (view_favorite_clicks or [])):
        return no_update, no_update
    
    # Extract city name from clicked button
    trigger = ctx.triggered[0]['prop_id']
    button_id = eval(trigger.split('.')[0])
    city = button_id['city']
    
    # Store city and redirect to home
    return city, "/"

# Suggestions callback - properly handles navigation states
@app.callback(
    Output('autofill-suggestions', 'children', allow_duplicate=True),
    [Input('city-input', 'value'),
     Input('weather-output', 'children'),  # Listen to weather output changes
     Input('recent-searches', 'data')],  # Input not State!
    [State('url', 'pathname'),
     State('selected-city-from-favorites', 'data')],  # Check if navigating from favorites
    prevent_initial_call=True
)


def show_autofill_suggestions_home(input_value, weather_output, recent_searches, current_path, favorites_navigation) -> html.Div:
    ctx = callback_context

    # If weather output changed, hide suggestions
    if 'weather-output.children' in [t['prop_id'] for t in ctx.triggered]:
        return html.Div()

    # If recent-searches updated (meaning search just completed), hide suggestions
    if 'recent-searches.data' in [t['prop_id'] for t in ctx.triggered]:
        return html.Div()  # Hide because search just completed


    # Don't show suggestions if we're in the middle of favorites navigation
    if favorites_navigation:
        return html.Div()

    if current_path != "/" and current_path is not None:
        return html.Div()
    
    ctx = callback_context
    if not ctx.triggered:
        return html.Div()
    
    trigger_id = ctx.triggered[0]['prop_id']
    recent_searches = recent_searches or []
    
    # Hide suggestions when user performs any action other than typing
    if any(x in trigger_id for x in ['search-btn', 'suggestion-click', 'welcome-city', 'n_submit']):
        return html.Div()
        
    
    # Case 1: Empty input - show recent searches and popular cities
    if 'city-input.value' in trigger_id:
        suggestions = []
    if not input_value or not input_value.strip():
        # Add recent searches section
        if recent_searches:
            suggestions.append(html.Div("Recent Searches", className="suggestions-header"))
            for city in recent_searches[-3:]:
                suggestions.append(
                    html.Div([
                        html.I(className="fas fa-clock me-2 text-muted"),
                        city
                    ], 
                    className="suggestion-item",
                    id={'type': 'suggestion-click', 'city': city},
                    n_clicks=0)
                )
        
        # Add popular cities section
        suggestions.append(html.Div("Popular Cities", className="suggestions-header"))
        for city in POPULAR_CITIES[:6]:
            suggestions.append(
                html.Div([
                    html.I(className="fas fa-map-marker-alt me-2 text-primary"),
                    city
                ], 
                className="suggestion-item",
                id={'type': 'suggestion-click', 'city': city},
                n_clicks=0)
            )
    
    # Case 2: User is typing - show matching suggestions
    elif len(input_value.strip()) >= 1:
        # Find matching recent searches
        matching_recent = [city for city in recent_searches if city.lower().startswith(input_value.lower())]
        if matching_recent:
            suggestions.append(html.Div("Recent Searches", className="suggestions-header"))
            for city in matching_recent:
                suggestions.append(
                    html.Div([
                        html.I(className="fas fa-clock me-2 text-muted"),
                        city
                    ], 
                    className="suggestion-item",
                    id={'type': 'suggestion-click', 'city': city},
                    n_clicks=0)
                )
        
        # Find matching popular cities (excluding those already in recent)
        matching_popular = [city for city in POPULAR_CITIES 
                           if city.lower().startswith(input_value.lower()) 
                           and city not in matching_recent]
        if matching_popular:
            suggestions.append(html.Div("Popular Cities", className="suggestions-header"))
            for city in matching_popular[:4]:
                suggestions.append(
                    html.Div([
                        html.I(className="fas fa-map-marker-alt me-2 text-primary"),
                        city
                    ], 
                    className="suggestion-item",
                    id={'type': 'suggestion-click', 'city': city},
                    n_clicks=0)
                )

    # Add manual dismiss option at the bottom
    if suggestions:
        suggestions.append(
            html.Div([
                html.I(className="fas fa-times me-2 text-muted"),
                "Hide suggestions"
            ], 
            className="suggestion-item text-muted",
            id={'type': 'hide-suggestions', 'action': 'hide'},
            n_clicks=0,
            style={'font-style': 'italic', 'font-size': '0.85rem'})
        )
    
    # Return dropdown if we have suggestions
    if suggestions:
        return html.Div(suggestions, className="suggestions-dropdown")
    return html.Div()

@app.callback(
    [Output('celsius-btn', 'color'),
     Output('fahrenheit-btn', 'color'),
     Output('current-units', 'data'),
     Output('weather-output', 'children', allow_duplicate=True)],
    [Input('celsius-btn', 'n_clicks'),
     Input('fahrenheit-btn', 'n_clicks')],
    [State('current-units', 'data'),
     State('current-city', 'data'),
     State('favorites-list', 'data'),
     State('url', 'pathname')],
    prevent_initial_call=True
)
def toggle_units_home(celsius_clicks: int, fahrenheit_clicks: int, current_units: str, 
                     current_city: str, favorites: List[str], current_path: str) -> Tuple[str, str, str, List]:
    """Handle temperature unit switching between Celsius and Fahrenheit - home page only."""
    
    # Only work on home page
    if current_path != "/" and current_path is not None:
        return no_update, no_update, no_update, no_update
        
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update
    
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger == 'celsius-btn' and celsius_clicks and celsius_clicks > 0:
        new_units = "metric"
        colors = ("primary", "outline-primary")
    elif trigger == 'fahrenheit-btn' and fahrenheit_clicks and fahrenheit_clicks > 0:
        new_units = "imperial"
        colors = ("outline-primary", "primary")
    else:
        return no_update, no_update, no_update, no_update
    
    # If we have a current city displayed, update its display with new units
    if current_city:
        weather_data, error = get_weather_data(current_city, new_units)
        if weather_data and not error:
            weather_display = create_weather_display(weather_data, new_units, favorites)
            return colors[0], colors[1], new_units, weather_display
    
    return colors[0], colors[1], new_units, no_update

# PAGE ROUTING
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname: str) -> html.Div:
    """Handle page routing between home and favorites pages."""
    if pathname == '/favorites':
        return create_favorites_page()
    else:
        return create_home_page()

# FAVORITES PAGE CALLBACKS
@app.callback(
    Output('favorites-page-content', 'children'),
    [Input('url', 'pathname'),
     Input('favorites-list', 'data')],
    prevent_initial_call=True
)
def display_favorites_page(pathname: str, favorites: List[str]) -> html.Div:
    """Display the favorites page content with current weather for each favorite city."""
    
    if pathname != '/favorites':
        return html.Div()
    
    favorites = favorites or []
    
    # Show message if no favorites
    if not favorites:
        return dbc.Alert([
            html.H4("No favorites yet!", className="alert-heading"),
            html.P("Add cities to your favorites from the weather search page."),
            html.Hr(),
            dbc.Button([
                html.I(className="fas fa-search me-2"),
                "Search Weather"
            ], href="/", color="primary", size="lg")
        ], color="info", className="text-center")
    
    # Create cards for each favorite city
    cards = []
    for city in favorites:
        
        weather_data, error = get_weather_data(city, "metric")
        
        if weather_data and not error:
            current = weather_data['current']
            temp = round(current['temp'])
            description = current['weather'][0]['description'].title()
            icon = current['weather'][0]['icon']
            location_info = weather_data['location_info']
            
            # Build location display text
            location_text = location_info['details']['city']
            if location_info['details']['country']:
                location_text += f", {location_info['details']['country']}"
            
            cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(city, className="card-title"),
                            html.P(location_text, className="text-muted small mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    html.H3(f"{temp}°C", className="text-primary mb-0 temperature-display"),
                                    html.P(description, className="small text-muted")
                                ], width=8),
                                dbc.Col([
                                    html.Img(
                                        src=f"http://openweathermap.org/img/w/{icon}.png",
                                        style={'width': '50px'}
                                    )
                                ], width=4)
                            ]),
                            html.Hr(),
                            dbc.ButtonGroup([
                                dbc.Button([
                                    html.I(className="fas fa-eye me-1"),
                                    "View Details"
                                ], 
                                id={'type': 'view-favorite-btn', 'city': city},
                                color="primary", 
                                size="sm",
                                n_clicks=0),
                                dbc.Button([
                                    html.I(className="fas fa-trash me-1"),
                                    "Remove"
                                ], 
                                id={'type': 'remove-favorite', 'city': city},
                                color="outline-danger", 
                                size="sm",
                                n_clicks=0)
                            ])
                        ])
                    ], className="h-100 weather-card")
                ], width=12, md=6, lg=4, className="mb-3")
            )
        else:
            # Show error card for cities with API issues
            cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(city, className="card-title"),
                            dbc.Alert("Unable to load weather data", color="warning", className="small"),
                            dbc.Button([
                                html.I(className="fas fa-trash me-1"),
                                "Remove"
                            ], 
                            id={'type': 'remove-favorite', 'city': city},
                            color="outline-danger", 
                            size="sm",
                            n_clicks=0)
                        ])
                    ], className="weather-card")
                ], width=12, md=6, lg=4, className="mb-3")
            )
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.P(f"You have {len(favorites)} favorite cities:", className="text-muted text-center mb-4")
            ])
        ]),
        dbc.Row(cards)
    ])

# FAVORITES MANAGEMENT CALLBACKS
@app.callback(
    [Output('favorites-list', 'data'),
     Output({'type': 'add-favorite-btn', 'city': ALL}, 'children'),
     Output({'type': 'add-favorite-btn', 'city': ALL}, 'color')],
    [Input({'type': 'add-favorite-btn', 'city': ALL}, 'n_clicks')],
    [State('favorites-list', 'data'),
     State({'type': 'add-favorite-btn', 'city': ALL}, 'id')],
    prevent_initial_call=True
)
def add_to_favorites(n_clicks_list: List[int], favorites: List[str], button_ids: List[Dict]) -> Tuple[List[str], List, List]:
    """Handle adding/removing cities from favorites list."""
    
    ctx = callback_context
    favorites = favorites or []
    
    # Handle case where there are no buttons on the page
    if not button_ids or not n_clicks_list:
        return favorites, [], []
    
    # Find which button was clicked
    if ctx.triggered and any((click or 0) > 0 for click in n_clicks_list):
        trigger = ctx.triggered[0]['prop_id']
        button_id = eval(trigger.split('.')[0])
        city = button_id['city']
        
        # Toggle favorite status
        if city in favorites:
            favorites.remove(city)
        else:
            favorites.append(city)
    
    # Update button appearance for all buttons
    updated_children = []
    updated_colors = []
    
    for btn_id in button_ids:
        btn_city = btn_id['city']
        if btn_city in favorites:
            updated_children.append([
                html.I(className="fas fa-heart-broken me-2"),
                "Remove from Favorites"
            ])
            updated_colors.append("danger")
        else:
            updated_children.append([
                html.I(className="fas fa-heart me-2"),
                "Add to Favorites"
            ])
            updated_colors.append("outline-primary")
    
    return favorites, updated_children, updated_colors

@app.callback(
    Output('favorites-list', 'data', allow_duplicate=True),
    [Input({'type': 'remove-favorite', 'city': ALL}, 'n_clicks')],
    [State('favorites-list', 'data')],
    prevent_initial_call=True
)
def remove_from_favorites(n_clicks_list: List[int], favorites: List[str]) -> List[str]:
    """Handle removing cities from favorites (from the favorites page)."""
    
    ctx = callback_context
    if ctx.triggered and any((click or 0) > 0 for click in (n_clicks_list or [])):
        trigger = ctx.triggered[0]['prop_id']
        button_id = eval(trigger.split('.')[0])
        city = button_id['city']
        
        favorites = favorites or []
        if city in favorites:
            favorites.remove(city)
        
        return favorites
    
    return favorites or []

# BETA FEATURES TRACKING
@app.callback(
    [Output('outfit-clicks', 'data'),
     Output('planner-clicks', 'data')],
    [Input({'type': 'outfit-interest-btn', 'city': ALL}, 'n_clicks'),
     Input({'type': 'planner-interest-btn', 'city': ALL}, 'n_clicks')],
    [State('outfit-clicks', 'data'),
     State('planner-clicks', 'data')],
    prevent_initial_call=True
)
def track_beta_interest(outfit_clicks: List[int], planner_clicks: List[int], 
                       current_outfit: int, current_planner: int) -> Tuple[int, int]:
    """Track user interest in beta features for analytics purposes."""
    
    ctx = callback_context
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id']
        if 'outfit-interest-btn' in trigger_id and any((click or 0) > 0 for click in (outfit_clicks or [])):
            return (current_outfit or 0) + 1, current_planner or 0
        elif 'planner-interest-btn' in trigger_id and any((click or 0) > 0 for click in (planner_clicks or [])):
            return current_outfit or 0, (current_planner or 0) + 1
    return current_outfit or 0, current_planner or 0

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    print("🌤️ SkyWise - Professional Weather Intelligence")
    print("🚀 Running at: http://127.0.0.1:8656")
    app.run_server(debug=True, port=8656)


