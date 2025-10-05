import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset from the Excel file
df = pd.read_excel('CTG_clean.xlsx')

# Display the first few rows to understand its structure
print(df.head())

# Check for missing values
print("Missing values in X:", df.isna().sum())

# Assuming the target variable is 'class' (adjust if necessary)
# Extract features (X) and targets (y)
X = df.drop(columns=['class'])  # Replace 'class' with the actual target column name
y = df['class']  # Replace 'class' with the actual target column name

# Handle missing values (if any)
X = X.dropna()
y = y[X.index]  # Keep the target aligned with the features

# Ensure all data in X is numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Convert target variable to numeric (if it's categorical)
y = y.astype('category').cat.codes

# Check for missing or infinite values in X and y
print("Missing values in X after conversion:", X.isna().sum())
print("Missing values in y after conversion:", y.isna().sum())

# Drop rows with NaN values created during conversion
X = X.dropna()
y = y[X.index]  # Align target with features

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ensure the shape of X_train and y_train is correct
print(X_train.shape)
print(y_train.shape)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ensure y_train is a 1D array (for Logistic Regression compatibility)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model and scaler for later use
joblib.dump(model, 'cardiotocography_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Print the training score
print("Training Accuracy: ", model.score(X_train, y_train))

# Evaluate on the test set
y_pred = model.predict(X_test)
print("Testing Accuracy: ", model.score(X_test, y_test))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
