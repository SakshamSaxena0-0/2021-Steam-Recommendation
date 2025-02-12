# Import Libraries #
import pandas as pd
import pickle
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Step 1: Load the Dataset Efficiently #
file_path = "path_to_dataset.csv"  # Change to your actual file path
chunk_size = 100000  # Load in chunks to prevent memory issues

chunks = []
for chunk in pd.read_csv(file_path, usecols=['app_id', 'app_name', 'review_id', 'author.playtime_forever'], chunksize=chunk_size):
    chunks.append(chunk)

    

# Merge first few chunks (adjust as needed) #
df = pd.concat(chunks[:20])  # Load first 500,000 rows
print(" Dataset Loaded Successfully!")

# Step 2: Preprocessing #
df.rename(columns={'author.playtime_forever': 'playtime_forever'}, inplace=True)

# Drop missing values #
df.dropna(subset=['playtime_forever'], inplace=True)

# Convert playtime to numeric #
df['playtime_forever'] = np.log1p(df['playtime_forever'])

# Remove zero playtime #
df = df[df['playtime_forever'] > 0]

# Step 3: Prepare Data for Surprise Library #
reader = Reader(rating_scale=(0, df['playtime_forever'].max()))
data = Dataset.load_from_df(df[['review_id', 'app_name', 'playtime_forever']], reader)

game_counts = df['app_name'].value_counts()
popular_games = game_counts[game_counts > len(df) * 0.3].index  # Games in >30% of users
df = df[~df['app_name'].isin(popular_games)]




#  Step 4: Train-Test Split #
trainset, testset = train_test_split(data, test_size=0.2)

# Step 5: Train SVD Model # 
model = SVD()
model.fit(trainset)

# Step 6: Evaluate Model #
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f" Model Trained! RMSE: {rmse:.2f}")

# Step 7: Save the Model #
with open("steam_recommendation_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model Saved Successfully as 'steam_recommendation_model.pkl'")
