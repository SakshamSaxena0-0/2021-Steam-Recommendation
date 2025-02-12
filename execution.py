# Import Libraries #
import pandas as pd
import pickle
from surprise import Dataset, Reader

# Load the Dataset Efficiently # 
file_path = "C:/Users/saksh/Downloads/steamR/steam_reviews.csv"  # Update with your file path
chunk_size = 100000  # Process in chunks

chunks = []
for chunk in pd.read_csv(file_path, usecols=['app_id', 'app_name', 'review_id', 'author.playtime_forever'], chunksize=chunk_size):
    chunks.append(chunk)

df = pd.concat(chunks[:5])  # Load first 500,000 rows
print(" Dataset Loaded Successfully!")

# Preprocessing #
df.rename(columns={'author.playtime_forever': 'playtime_forever'}, inplace=True)

# Apply a Minimum Playtime Filter #
df = df[df['playtime_forever'] > 60]  # Ignore users with < 1 hour playtime

# Get Unique Users #
unique_users = df['review_id'].unique()
print(f"🔹 Total Unique Users: {len(unique_users)}")

# Ask for User Input #
user_id = input(" Enter User ID: ")
if user_id.isdigit():
    user_id = int(user_id)
    if user_id not in unique_users:
        print(" User ID not found in dataset! Please try another.")
        exit()
else:
    print(" Invalid input! Please enter a numeric User ID.")
    exit()

print(f" Fetching recommendations for User ID: {user_id}...")

# Load Trained Model #
with open("steam_recommendation_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

print("Model Loaded Successfully!")

#  Recommendation Function #
def recommend_games_for_user(user_id, num_recommendations=5):
    all_games = df['app_name'].unique()
    user_games = df[df['review_id'] == user_id]['app_name'].tolist()

    # Predict scores for unseen games
    predictions = [(game, model.predict(user_id, game).est) for game in all_games if game not in user_games]
    
    # Sort and return top recommendations #
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    return [game for game, _ in recommendations]


# Get Recommendations #
recommended_games = recommend_games_for_user(user_id)

print(f"\n Recommended games for User ID {user_id}: {recommended_games}")

