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

# Check for missing values in the dataset
print("Missing values in X:", df.isna().sum())

# Assuming the target variable is 'class' (adjust if necessary)
# Extract features (X) and targets (y)
X = df.drop(columns=['class'])  # Replace 'class' with the actual target column name
y = df['class']  # Replace 'class' with the actual target column name

# Handle missing values (if any)
X = X.dropna()  # Drop rows with missing values in features
y = y[X.index]  # Align the target with the remaining rows in X

# Ensure all data in X is numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Convert target variable to numeric (if it's categorical)
y = y.astype('category').cat.codes

# Check for missing values in features and target after cleaning
print("Missing values in X after conversion:", X.isna().sum())
print("Missing values in y after conversion:", y.isna().sum())

# Drop rows with NaN values created during conversion (if necessary)
X = X.dropna()
y = y[X.index]  # Align target with features

# Ensure that the number of samples in X and y match
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Ensure X and y have matching indices
print("X and y indices match?", (X.index == y.index).all())

# Align X and y to ensure matching indices
X = X.loc[y.index]

# Check if X or y are empty
print(f"Is X empty? {X.empty}")
print(f"Is y empty? {y.empty}")

# Check the first few entries of X and y
print("First few rows of X:", X.head())
print("First few values of y:", y.head())

# Reset index for both X and y
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Check the shapes after resetting the index
print("Shape of X after resetting index:", X.shape)
print("Shape of y after resetting index:", y.shape)

# Check for NaN values in X and y
print("Missing values in X:", X.isna().sum())
print("Missing values in y:", y.isna().sum())

# Print the number of rows in X and y
print(f"Number of rows in X: {len(X)}")
print(f"Number of rows in y: {len(y)}")

# Attempt to split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the shapes after splitting
print(f"Training set X shape: {X_train.shape}")
print(f"Training set y shape: {y_train.shape}")

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
