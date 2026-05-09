# 🌤️ Live Weather API Setup Guide

## Getting Your Free OpenWeatherMap API Key

The clothing recommender system now integrates with **OpenWeatherMap** to fetch real-time weather data.

### Step 1: Get a Free API Key

1. Visit [https://openweathermap.org/api](https://openweathermap.org/api)
2. Click on "Sign Up" and create a free account
3. Go to your Account → API Keys
4. Copy your default API key (it takes ~10 minutes to activate)

### Step 2: Configure Your API Key

Choose one of these methods:

#### Method 1: Environment Variable (Recommended)

**Windows:**
```powershell
$env:OPENWEATHER_API_KEY = "your_api_key_here"
python app.py
```

**Mac/Linux:**
```bash
export OPENWEATHER_API_KEY="your_api_key_here"
python app.py
```

#### Method 2: Pass to ClothingRecommender

In `app.py`, modify the initialization:
```python
recommender = ClothingRecommender(weather_api_key="your_api_key_here")
```

#### Method 3: Store in .env File

1. Create a `.env` file in the project root:
```
OPENWEATHER_API_KEY=your_api_key_here
```

2. Install python-dotenv:
```bash
pip install python-dotenv
```

3. Update `app.py`:
```python
from dotenv import load_dotenv
load_dotenv()

recommender = ClothingRecommender(weather_api_key=os.environ.get('OPENWEATHER_API_KEY'))
```

## How It Works

Once configured, users can:

1. **Select a city** from the dropdown
2. **Click "Fetch Weather"** button
3. Temperature and weather condition auto-fill
4. **Submit** for outfit recommendations based on live weather

## API Endpoints

### 1. Fetch Weather Only
```
GET /api/weather?city=London
```

**Response:**
```json
{
  "success": true,
  "temperature": 15.2,
  "weather_condition": "rainy",
  "weather_description": "light rain",
  "city": "London",
  "country": "GB"
}
```

### 2. Get Recommendations with Live Weather
```
GET /api/weather-recommendation?city=London&occasion=casual
```

**Response:**
```json
{
  "success": true,
  "city": "London",
  "temperature": 15.2,
  "weather": "light rain",
  "outfits": [...],
  "advice": [...]
}
```

## Troubleshooting

### "API key not configured" Error
- Environment variable not set
- API key hasn't activated yet (wait 10 minutes after signing up)
- Typo in API key

### "Invalid API key" Error
- Copy the entire key correctly from OpenWeatherMap
- Make sure it's the "Default" API key (not a personal one)

### "City not found" Error
- Check city spelling
- Use English city names
- Try major cities first to test

## Free Tier Limits

OpenWeatherMap Free API allows:
- **Calls/minute:** 60
- **Calls/day:** 1,000,000
- **Data:** Current weather, 5-day forecast

Perfect for a personal clothing recommender!

---

**Questions?** Visit [OpenWeatherMap Documentation](https://openweathermap.org/api/current-weather)
