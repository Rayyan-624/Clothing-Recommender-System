# ⚡ Quick Start: Live Weather Integration

## 30-Second Setup

### Step 1: Get API Key (2 minutes)
```
1. Go to: https://openweathermap.org/api
2. Click "Sign Up" (free)
3. Go to Account > API Keys
4. Copy your key (it says "Default")
⏳ Wait 10 minutes for activation
```

### Step 2: Set Environment Variable (10 seconds)

**Windows (PowerShell):**
```powershell
$env:OPENWEATHER_API_KEY = "paste_your_key_here"
python app.py
```

**Mac/Linux:**
```bash
export OPENWEATHER_API_KEY="paste_your_key_here"
python app.py
```

### Step 3: Test (30 seconds)
1. Open http://localhost:5000
2. Select a city from dropdown
3. Click green "☁️ Fetch Weather" button
4. Watch form auto-fill with live weather!
5. Click "Get Outfit Recommendations"
6. See "✓ Live API" badge on results page

---

## That's It! 🎉

Your app now:
- ✅ Fetches real-time weather
- ✅ Shows live data badges
- ✅ Works without API key (manual mode)
- ✅ Has 3 REST API endpoints

---

## API Endpoints (for integration/testing)

```bash
# Get weather for any city
curl "http://localhost:5000/api/weather?city=London"

# Get recommendations + weather in one call
curl "http://localhost:5000/api/weather-recommendation?city=London&occasion=casual"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "API key not configured" | Set `OPENWEATHER_API_KEY` environment variable |
| "Invalid API key" | Copy exact key from OpenWeatherMap, wait 10 mins |
| "City not found" | Try a major city (London, Tokyo, New York) |
| Fetch button doesn't work | Check browser console for errors |

---

## Files to Check

- `WEATHER_API_SETUP.md` - Detailed setup guide
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `test_weather_api.py` - Test the integration

---

**Ready to impress with live weather? You're all set!** ✨
