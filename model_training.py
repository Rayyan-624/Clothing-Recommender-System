import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

class ClothingModelTrainer:
    def __init__(self, dataset_path='data/clothing_dataset.csv'):
        self.df = pd.read_csv(dataset_path)
        self.label_encoders = {}
        self.model = None
    
    def prepare_features(self):
        """Prepare features for ML model"""
        # Create expanded dataset with synthetic weather-occasion combinations
        expanded_data = []
        
        weather_types = ['sunny', 'cloudy', 'rainy', 'snowy', 'hot', 'cold', 'mild']
        occasions = ['casual', 'business', 'formal', 'sports', 'date', 'outdoor']
        temperatures = list(range(-5, 36, 5))  # -5 to 35°C
        
        # For each combination, find suitable items
        for temp in temperatures:
            for weather in weather_types:
                for occasion in occasions:
                    # Simplified logic to label items as suitable/not suitable
                    for _, item in self.df.iterrows():
                        # Feature vector
                        features = [
                            temp,
                            self._encode_weather(weather),
                            self._encode_occasion(occasion),
                            item['warmth_level'],
                            item['formality'],
                            item['min_temp'],
                            item['max_temp']
                        ]
                        
                        # Label: 1 if suitable, 0 otherwise
                        label = self._is_suitable(item, temp, weather, occasion)
                        
                        expanded_data.append(features + [label])
        
        columns = ['temperature', 'weather', 'occasion', 'warmth', 
                  'formality', 'min_temp', 'max_temp', 'suitable']
        return pd.DataFrame(expanded_data, columns=columns)
    
    def _encode_weather(self, weather):
        """Encode weather type numerically"""
        mapping = {'sunny': 0, 'cloudy': 1, 'rainy': 2, 'snowy': 3, 
                  'hot': 4, 'cold': 5, 'mild': 6}
        return mapping.get(weather, 1)
    
    def _encode_occasion(self, occasion):
        """Encode occasion type numerically"""
        mapping = {'casual': 0, 'business': 1, 'formal': 2, 
                  'sports': 3, 'date': 4, 'outdoor': 5}
        return mapping.get(occasion, 0)
    
    def _is_suitable(self, item, temperature, weather, occasion):
        """Determine if item is suitable for given conditions"""
        # Check temperature range
        if not (item['min_temp'] <= temperature <= item['max_temp']):
            return 0
        
        # Check weather compatibility
        weather_tags = item['weather_tags'].split(',')
        if weather not in weather_tags and 'all' not in weather_tags:
            return 0
        
        # Check occasion compatibility
        occasion_tags = item['occasion_tags'].split(',')
        if occasion not in occasion_tags and 'all' not in occasion_tags:
            return 0
        
        return 1
    
    def train_model(self):
        """Train Random Forest classifier"""
        print("Preparing training data...")
        data = self.prepare_features()
        
        # Split features and labels
        X = data.drop('suitable', axis=1)
        y = data['suitable']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train Random Forest
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        joblib.dump(self.model, 'models/clothing_model.pkl')
        print("Model saved to models/clothing_model.pkl")
        
        return accuracy
    
    def predict_suitability(self, temperature, weather, occasion, item_features):
        """Predict if an item is suitable for given conditions"""
        if self.model is None:
            # Load trained model
            self.model = joblib.load('models/clothing_model.pkl')
        
        # Prepare input features
        features = np.array([[
            temperature,
            self._encode_weather(weather),
            self._encode_occasion(occasion),
            item_features['warmth_level'],
            item_features['formality'],
            item_features['min_temp'],
            item_features['max_temp']
        ]])
        
        prediction = self.model.predict(features)
        probability = self.model.predict_proba(features)[0][1]
        
        return bool(prediction[0]), probability

# Train and save model
if __name__ == "__main__":
    trainer = ClothingModelTrainer()
    accuracy = trainer.train_model()
    print(f"\n✅ Model training complete with {accuracy:.2%} accuracy")