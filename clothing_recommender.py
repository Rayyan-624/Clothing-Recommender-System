import pandas as pd
import json
import random
from typing import List, Dict, Tuple

class ClothingRecommender:
    def __init__(self, dataset_path='data/clothing_dataset.csv'):
        """Initialize the recommender system with clothing dataset"""
        self.df = pd.read_csv(dataset_path)
        self.weather_mapping = self._load_weather_mapping()
        self.occasion_formality = {
            'casual': 1,
            'business': 4,
            'formal': 5,
            'sports': 1,
            'date': 3,
            'outdoor': 2,
            'beach': 1,
            'party': 4
        }
    
    def _load_weather_mapping(self):
        """Load weather condition mappings"""
        with open('data/weather_codes.json', 'r') as f:
            return json.load(f)
    
    def get_weather_type(self, temperature: float, condition: str) -> str:
        """Determine weather type based on temperature and condition"""
        condition = condition.lower()
        
        # Check specific conditions first
        if 'rain' in condition or 'drizzle' in condition:
            return 'rainy'
        elif 'snow' in condition:
            return 'snowy'
        elif 'clear' in condition:
            return 'sunny'
        elif 'cloud' in condition:
            return 'cloudy'
        elif 'wind' in condition:
            return 'windy'
        
        # Fallback to temperature-based classification
        if temperature >= 28:
            return 'hot'
        elif temperature <= 10:
            return 'cold'
        else:
            return 'mild'
    
    def calculate_item_score(self, item: pd.Series, weather_type: str, 
                           temperature: float, occasion: str) -> float:
        """Calculate suitability score for a clothing item"""
        score = 0
        
        # 1. Temperature suitability (40%)
        if item['min_temp'] <= temperature <= item['max_temp']:
            temp_score = 40
        else:
            # Penalize based on deviation
            deviation = min(abs(temperature - item['min_temp']), 
                          abs(temperature - item['max_temp']))
            temp_score = max(0, 40 - deviation * 2)
        score += temp_score
        
        # 2. Weather condition match (30%)
        weather_tags = item['weather_tags'].split(',')
        if weather_type in weather_tags:
            score += 30
        
        # 3. Occasion formality match (30%)
        occasion_tags = item['occasion_tags'].split(',')
        target_formality = self.occasion_formality.get(occasion, 2)
        formality_diff = abs(item['formality'] - target_formality)
        formality_score = max(0, 30 - formality_diff * 6)
        score += formality_score
        
        return score
    
    def recommend_outfit(self, temperature: float, weather_condition: str, 
                        occasion: str, num_recommendations: int = 3) -> List[Dict]:
        """
        Recommend complete outfits based on weather and occasion
        
        Args:
            temperature: Current temperature in Celsius
            weather_condition: Weather description (e.g., 'clear', 'rainy')
            occasion: User's occasion/activity
            num_recommendations: Number of outfit recommendations to return
        
        Returns:
            List of recommended outfits with items and scores
        """
        weather_type = self.get_weather_type(temperature, weather_condition)
        
        # Calculate scores for all items
        scored_items = []
        for _, item in self.df.iterrows():
            score = self.calculate_item_score(item, weather_type, temperature, occasion)
            scored_items.append((item.to_dict(), score))
        
        # Sort by score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Group into outfits (top, bottom, footwear, outerwear, accessory)
        outfits = []
        used_items = set()
        
        for i in range(min(num_recommendations * 5, len(scored_items))):
            item, score = scored_items[i]
            
            if item['item_id'] in used_items:
                continue
                
            # Try to build a complete outfit starting with this item
            outfit = self._build_outfit_around_item(item, scored_items, used_items, 
                                                   temperature, weather_type, occasion)
            
            if outfit and len(outfit['pieces']) >= 3:  # At least top, bottom, footwear
                outfits.append(outfit)
                if len(outfits) >= num_recommendations:
                    break
        
        return outfits
    
    def _build_outfit_around_item(self, main_item: Dict, scored_items: List[Tuple], 
                                 used_items: set, temperature: float, 
                                 weather_type: str, occasion: str) -> Dict:
        """Build a complete outfit around a main item"""
        outfit = {
            'pieces': [main_item],
            'categories': {main_item['category']},
            'total_score': scored_items[0][1],
            'weather_type': weather_type
        }
        
        # Required categories for a complete outfit
        required_cats = {'Top', 'Bottom', 'Footwear'}
        current_cats = {main_item['category']}
        
        # Add complementary items
        for item, score in scored_items[1:]:
            if item['item_id'] in used_items:
                continue
                
            # Check if item complements the outfit
            if (item['category'] not in current_cats and 
                item['category'] in ['Top', 'Bottom', 'Footwear', 'Outerwear', 'Accessory']):
                
                # Check compatibility
                if self._items_compatible(main_item, item, temperature, weather_type):
                    outfit['pieces'].append(item)
                    outfit['categories'].add(item['category'])
                    outfit['total_score'] += score
                    current_cats.add(item['category'])
                    used_items.add(item['item_id'])
            
            # Stop when we have a complete outfit
            if required_cats.issubset(current_cats) and len(outfit['pieces']) >= 4:
                break
        
        # Calculate average score
        outfit['avg_score'] = outfit['total_score'] / len(outfit['pieces'])
        
        # Convert set to list for JSON serialization
        outfit['categories'] = list(outfit['categories'])
        
        return outfit
    
    def _items_compatible(self, item1: Dict, item2: Dict, temperature: float, 
                         weather_type: str) -> bool:
        """Check if two clothing items are compatible"""
        # Don't pair two items of same main category
        if item1['category'] == item2['category'] and item1['category'] in ['Top', 'Bottom']:
            return False
        
        # Check temperature compatibility
        temp_range1 = (item1['min_temp'], item1['max_temp'])
        temp_range2 = (item2['min_temp'], item2['max_temp'])
        
        # Check if temperature ranges overlap
        if temp_range1[1] < temp_range2[0] or temp_range2[1] < temp_range1[0]:
            return False
        
        # Check weather compatibility
        weather_tags1 = set(item1['weather_tags'].split(','))
        weather_tags2 = set(item2['weather_tags'].split(','))
        
        # Allow if either item has "all" tag, or if they share common tags
        # This makes the system more flexible for weather types like "cloudy"
        if 'all' in weather_tags1 or 'all' in weather_tags2:
            return True
        if weather_type in weather_tags1 or weather_type in weather_tags2:
            return True
        if len(weather_tags1.intersection(weather_tags2)) > 0:
            return True
        
        return False
    
    def get_weather_advice(self, temperature: float, weather_type: str) -> str:
        """Generate weather-specific clothing advice"""
        advice = []
        
        if weather_type == 'rainy':
            advice.append("☔ Bring a raincoat or umbrella")
            advice.append("💧 Waterproof footwear recommended")
        elif weather_type == 'snowy':
            advice.append("❄️ Wear thermal layers")
            advice.append("🧤 Don't forget gloves and warm hat")
        elif weather_type == 'sunny' or weather_type == 'hot':
            advice.append("☀️ Wear light, breathable fabrics")
            advice.append("🕶️ Sunglasses are recommended")
        elif weather_type == 'cold':
            advice.append("🧥 Layer up with warm clothing")
            advice.append("🧣 Scarf and gloves will help keep you warm")
        elif weather_type == 'windy':
            advice.append("💨 Wind-resistant jacket recommended")
            advice.append("🎩 Secure hat or consider hood")
        
        # Temperature-specific advice
        if temperature > 25:
            advice.append("🌡️ Stay hydrated in this heat")
        elif temperature < 5:
            advice.append("🧊 Protect exposed skin from cold")
        
        return advice