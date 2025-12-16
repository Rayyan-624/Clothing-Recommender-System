from flask import Flask, render_template, request, jsonify
import json
import os
from clothing_recommender import ClothingRecommender
from model_training import ClothingModelTrainer

app = Flask(__name__)

# Initialize components
recommender = ClothingRecommender()
model_trainer = ClothingModelTrainer()

# Sample cities for demo
CITIES = {
    'New York': {'lat': 40.7128, 'lon': -74.0060},
    'London': {'lat': 51.5074, 'lon': -0.1278},
    'Tokyo': {'lat': 35.6762, 'lon': 139.6503},
    'Sydney': {'lat': -33.8688, 'lon': 151.2093},
    'Dubai': {'lat': 25.2048, 'lon': 55.2708},
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
    'Toronto': {'lat': 43.6532, 'lon': -79.3832},
    'Berlin': {'lat': 52.5200, 'lon': 13.4050}
}

# Sample weather conditions
WEATHER_CONDITIONS = [
    'clear', 'cloudy', 'rainy', 'snowy', 'windy', 
    'partly cloudy', 'thunderstorm', 'foggy'
]

# Occasions
OCCASIONS = [
    'casual', 'business', 'formal', 'sports', 
    'date', 'outdoor', 'beach', 'party'
]

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', 
                         cities=list(CITIES.keys()),
                         weather_conditions=WEATHER_CONDITIONS,
                         occasions=OCCASIONS)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Generate clothing recommendations"""
    try:
        # Get form data
        temperature = float(request.form.get('temperature', 20))
        weather_condition = request.form.get('weather_condition', 'clear')
        occasion = request.form.get('occasion', 'casual')
        city = request.form.get('city', '')
        
        # Generate recommendations
        outfits = recommender.recommend_outfit(
            temperature=temperature,
            weather_condition=weather_condition,
            occasion=occasion,
            num_recommendations=3
        )
        
        # Get weather advice
        weather_type = recommender.get_weather_type(temperature, weather_condition)
        advice = recommender.get_weather_advice(temperature, weather_type)
        
        # Prepare response
        response = {
            'success': True,
            'input': {
                'temperature': temperature,
                'weather_condition': weather_condition,
                'weather_type': weather_type,
                'occasion': occasion,
                'city': city
            },
            'outfits': outfits,
            'advice': advice,
            'weather_type': weather_type
        }
        
        return render_template('results.html', **response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    """API endpoint for recommendations"""
    try:
        temperature = float(request.args.get('temp', 20))
        weather = request.args.get('weather', 'clear')
        occasion = request.args.get('occasion', 'casual')
        
        outfits = recommender.recommend_outfit(temperature, weather, occasion, 2)
        
        return jsonify({
            'success': True,
            'recommendations': outfits
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def predict():
    """ML model prediction endpoint"""
    try:
        data = request.json
        
        prediction, probability = model_trainer.predict_suitability(
            temperature=data['temperature'],
            weather=data['weather'],
            occasion=data['occasion'],
            item_features=data['item_features']
        )
        
        return jsonify({
            'success': True,
            'suitable': prediction,
            'confidence': float(probability),
            'message': 'Suitable' if prediction else 'Not suitable'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/dashboard')
def dashboard():
    """Admin dashboard"""
    # Load dataset stats
    df = recommender.df
    stats = {
        'total_items': len(df),
        'categories': df['category'].value_counts().to_dict(),
        'avg_warmth': df['warmth_level'].mean(),
        'avg_formality': df['formality'].mean()
    }
    
    return render_template('dashboard.html', stats=stats)

@app.route('/api-docs')
def api_docs():
    """API documentation page"""
    return render_template('api_docs.html')

@app.route('/help')
def help_page():
    """Help and FAQ page"""
    return render_template('help.html')


if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=True)