#!/usr/bin/env python
"""Quick test of the live weather API integration"""

from clothing_recommender import ClothingRecommender

print("Testing Live Weather API Integration...")
print("=" * 50)

recommender = ClothingRecommender()

# Test 1: Fetch weather for London
print("\n1. Testing fetch_live_weather('London')...")
result = recommender.fetch_live_weather('London')
if result['success']:
    print(f"   ✓ Success!")
    print(f"   City: {result['city']}, {result['country']}")
    print(f"   Temperature: {result['temperature']}°C")
    print(f"   Weather: {result['weather_description']}")
else:
    print(f"   ✗ Error: {result['error']}")

# Test 2: Fetch weather for Tokyo
print("\n2. Testing fetch_live_weather('Tokyo')...")
result = recommender.fetch_live_weather('Tokyo')
if result['success']:
    print(f"   ✓ Success!")
    print(f"   City: {result['city']}, {result['country']}")
    print(f"   Temperature: {result['temperature']}°C")
    print(f"   Weather: {result['weather_description']}")
else:
    print(f"   ✗ Error: {result['error']}")

# Test 3: Test weather condition mapping
print("\n3. Testing weather condition mapping...")
weather_tests = [
    ('Clear', 'sunny', 'clear'),
    ('Rain', 'light rain', 'rainy'),
    ('Snow', 'light snow', 'snowy'),
    ('Clouds', 'overcast clouds', 'cloudy'),
]

for main, desc, expected in weather_tests:
    result = recommender._map_weather_condition(main.lower(), desc.lower())
    status = "✓" if result == expected else "✗"
    print(f"   {status} {main} → {result} (expected: {expected})")

print("\n" + "=" * 50)
print("Live Weather API Integration Test Complete!")
