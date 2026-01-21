import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 1. Load the Titanic Dataset
# We load directly from seaborn to make this script plug-and-play
df = sns.load_dataset('titanic')

# 2. Feature Selection & Preprocessing
# We select exactly 5 features + target
# Features: Pclass (class), Sex, Age, SibSp (siblings/spouses), Fare
# Target: survived

selected_features = ['pclass', 'sex', 'age', 'sibsp', 'fare']
target = 'survived'

# Filter the dataframe
df = df[selected_features + [target]].copy()

# A. Handle Missing Values
# Age usually has missing values. We fill them with the median.
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())

# B. Encode Categorical Variables
# 'sex' is male/female. We need to convert it to numbers.
# Let's use a simple mapping for clarity: Male=0, Female=1
le_sex = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])
# Note: Keep track of this mapping! (male -> 1, female -> 0 usually, check le_sex.classes_)
# Actually, let's force a manual map to be safe for the App later:
# map: {'male': 0, 'female': 1}
# (Re-loading to ensure mapping is explicit for the web app)
df = sns.load_dataset('titanic')[selected_features + [target]].copy()
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# 3. Data Splitting
X = df[selected_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling (Optional for Random Forest, but good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the Model
# Algorithm: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training Model...")
model.fit(X_train_scaled, y_train)

# 6. Evaluate the Model
y_pred = model.predict(X_test_scaled)

print("--- Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 7. Save the Model and Scaler
# We save both to ensure the App can scale inputs exactly like the training data
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump({'model': model, 'scaler': scaler}, 'model/titanic_survival_model.pkl')
print("Model and Scaler saved successfully to model/titanic_survival_model.pkl")

# 8. Reload Demonstration
loaded_package = joblib.load('model/titanic_survival_model.pkl')
loaded_model = loaded_package['model']
print("Model reloaded successfully.")