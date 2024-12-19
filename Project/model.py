import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
# You imported DecisionTreeClassifier, but are not using it, so it's not necessary unless you want to compare models.
# from sklearn.tree import DecisionTreeClassifier 

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Check for any missing values or outliers in the dataset.
print(df.isnull().sum())  # To check for missing values in the dataset

# Split the dataset into features and labels
X = df.iloc[:, :-1]  # All columns except the last one (features)
y = df.iloc[:, -1]   # The last column (target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier (you can adjust parameters as needed)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file
pickle.dump(model, open("model.pkl", "wb"))

# To check if model is trained successfully, you can print the model's accuracy on the test set
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
