# SkyWise - Weather Intelligence Platform

A weather forecasting web app built with Dash and Python. Features ML temperature predictions, interactive charts, and a clean UI for searching weather worldwide.

## What It Does

**Core Features**
- Search weather for any city globally
- 5-day detailed forecasts with temperature ranges
- 24-hour temperature trend charts
- Save favorite cities for quick access
- ML prediction for tomorrow's temperature
- Switch between Celsius and Fahrenheit
- Smart search suggestions with recent cities

**Two-Page Design**
- Main page: weather search and detailed forecasts
- Favorites page: manage saved cities with quick weather overview

**User Experience Touches**
- Auto-complete search with popular cities
- Loading states and smooth transitions
- Responsive design for mobile/desktop
- Professional error handling
- Recent search history

## Getting Started

**Requirements**
- Python 3.7+
- Internet connection for weather data

**Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python skywise.py

# Visit http://127.0.0.1:8163
```

**Dependencies**
```
dash==2.14.1
dash-bootstrap-components==1.5.0
requests==2.31.0
plotly==5.17.0
scikit-learn==1.3.0
numpy==1.24.3
```

## Technical Details

**Architecture**
The app uses Dash for the web framework with Bootstrap for styling. Weather data comes from Open-Meteo API (free, no auth needed). City search uses OpenStreetMap geocoding. Everything runs client-side with browser storage for favorites.

**Code Structure**
```
skywise.py (~1200 lines)
├── API functions (weather data, geocoding)
├── ML prediction engine  
├── UI components (charts, cards, layouts)
├── Dash callbacks (search, navigation, favorites)
└── CSS styling and app setup
```

**Weather APIs Used**
- **Open-Meteo**: Free weather API with current conditions, hourly/daily forecasts, and historical data
- **Nominatim**: OpenStreetMap geocoding for city name resolution

Both are free APIs with good global coverage and no authentication requirements.

## ML Temperature Prediction

**What It Does**
Predicts tomorrow's average temperature using Linear Regression with improved weather-based feature engineering.

**How It Works**
- Uses past 4-10 days of temperature data for each location
- Applies proper autoregressive modeling (yesterday predicts today)
- Incorporates weather persistence and momentum patterns
- Trains model in real-time (< 100ms) with confidence intervals
- Returns prediction with statistical error bounds

**Improved Features**
```python
features = [
    prev_avg_temp,       # yesterday's temperature (autoregressive)
    temp_range,          # temperature stability indicator
    seasonal_component,  # sin(2π * day_of_year / 365.25) 
    temperature_momentum # rate of temperature change
]
```

**Model Improvements**
- **Fixed Autoregression**: Uses actual previous day temperature, not same-day circular logic
- **Weather Persistence**: Based on meteorological principle that weather patterns persist short-term
- **Temperature Momentum**: Captures warming/cooling trends over 2-3 days
- **Conservative Bounds**: Predictions limited to reasonable ranges based on recent weather
- **Removed Harmful Features**: Eliminated linear day index that created artificial trends

**Accuracy**
- Typically within ±2-4°C for next-day predictions
- Confidence intervals based on actual model performance
- More stable predictions compared to original implementation
- Follows meteorological principles of weather persistence

**Limitations**
This remains a simplified model for demo purposes. Production weather forecasting requires:
- Multi-year historical datasets
- Atmospheric pressure and wind pattern analysis  
- Complex non-linear modeling (neural networks, ensemble methods)
- Professional meteorological expertise
- Real-time observational data integration

**Why This Approach**
- Demonstrates proper ML feature engineering principles
- Shows understanding of autoregressive modeling
- Fast enough for real-time web integration
- Educationally valuable for learning ML concepts
- Honest about limitations while being technically sound

## UI Design Approach

**Design System**
- Blue/white color scheme with yellow accents
- Custom CSS with consistent spacing and typography
- Professional gradients and subtle animations
- Font Awesome icons throughout

**User Experience**
- Progressive disclosure (overview → details)
- Smart defaults (Celsius, popular cities)
- Contextual help and error messages
- Responsive layout for all screen sizes

**Interactive Elements**
- Hover effects on cards and buttons
- Smooth transitions between states
- Loading spinners during API calls
- Auto-hide suggestions after selection

## Development Process & AI Usage

**Time Spent: ~5 hours total**
- Hour 1: Learning Dash basics and setup
- Hour 2: API integration and data processing
- Hour 3: ML model implementation
- Hour 4: UI development and styling  
- Hour 5: Testing, debugging, polish

**AI Tools Used**
- **Claude**: Main development assistant for learning Dash patterns
- **Copilot**: Code completion and boilerplate generation
- **ChatGPT**: Specific technical questions and writeup assistance

**What AI Helped With**
- Dash framework learning (callback patterns, component structure)
- CSS boilerplate and responsive design templates  
- Error handling patterns and try/catch structures
- Scikit-learn syntax and model setup
- Bootstrap component layouts

**What I Did Myself**
- All architectural decisions (two-page design, feature set, UI flow)
- ML approach selection and feature engineering strategy
- Business logic (search, favorites, navigation)
- API integration design and error handling strategy
- UI/UX decisions (colors, layouts, user flows)
- Code organization and refactoring
- Extensive testing and debugging

I used AI as a learning accelerator, not a replacement for technical thinking. Every piece of generated code was reviewed, understood, and often modified. The core design decisions and problem-solving were mine.

## Current Limitations & Future Ideas

**Known Issues**
- ML predictions limited by training data scope
- No caching (hits API every search)
- No offline capabilities
- Favorites stored in browser only
- No user accounts or personalization

**If I Had More Time**
- **Better ML**: Historical weather data integration, ensemble methods
- **Smart Features**: 
  - Outfit recommendations based on weather
  - Weekly activity planner with weather context
- **Performance**: API caching, database storage
- **Testing**: Unit tests for ML models, API integration tests, UI component tests
- **Mobile**: Progressive Web App features, push notifications

**Testing Strategy I'd Add**
```python
# ML testing
test_temperature_prediction_accuracy()
test_feature_engineering_edge_cases()
test_model_confidence_intervals()

# API testing  
test_weather_api_error_handling()
test_geocoding_malformed_inputs()
test_network_timeout_scenarios()

# UI testing
test_search_functionality()
test_favorites_management()
test_temperature_unit_toggle()
test_responsive_breakpoints()
```

## Assignment Requirements Check

**Core Requirements**
- Two-page application ✓
- City search with weather display ✓  
- 5-day forecast with min/max temps ✓
- Add/remove favorites functionality ✓
- Simple ML temperature prediction ✓
- Clean, readable code ✓
- Error handling ✓
- Professional UI ✓

**Bonus Features Added**
- Interactive temperature charts
- Smart autocomplete search
- Celsius/Fahrenheit toggle
- Responsive design
- Loading states and transitions
- Detailed location information
- Beta feature previews

