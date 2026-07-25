import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'titanic_model.pkl')
data_path = os.path.join(current_dir, 'Titanic_train.csv')

# Function to train model
def train_model():
    """Train the model from scratch using Titanic training data"""
    try:
        train_df = pd.read_csv(data_path)
        
        # Handle missing values
        train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
        train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
        
        # Drop unnecessary columns
        df = train_df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)
        
        # Encode categorical variables
        le = LabelEncoder()
        df['Sex'] = le.fit_transform(df['Sex'])  # male: 1, female: 0
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
        
        # Prepare data
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        
        # Split and train
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model, True
    except Exception as e:
        return None, False

# Load or train the model
model = None
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except (FileNotFoundError, pickle.UnpicklingError, EOFError):
    st.warning("⚠️ Model file is missing or corrupted. Training a new model...")
    model, success = train_model()
    if success:
        st.success("✅ Model trained successfully!")
    else:
        st.error(f"❌ Error: Could not train model. Please ensure 'Titanic_train.csv' exists at {data_path}")
        st.stop()

if model is None:
    st.error("❌ Error: Could not load or train model. Please check your files.")
    st.stop()

# Streamlit App Title
st.title('⚓ Titanic Survival Predictor')
st.write('Enter the passenger details to predict survival probability.')

# Input fields for user data
st.sidebar.header('Passenger Information')
pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
sex = st.sidebar.selectbox('Sex', ['male', 'female'])
age = st.sidebar.slider('Age', 0.42, 80.0, 25.0)
sibsp = st.sidebar.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
parch = st.sidebar.slider('Number of Parents/Children Aboard', 0, 6, 0)
fare = st.sidebar.slider('Fare (£)', 0.0, 512.3292, 30.0)
embarked = st.sidebar.selectbox('Port of Embarkation', ['C', 'Q', 'S'], help="C = Cherbourg, Q = Queenstown, S = Southampton")

# Preprocess input data to match model training format
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    # Create a DataFrame from input
    data = {
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    }
    input_df = pd.DataFrame(data)

    # Encode 'Sex' (male: 1, female: 0, as per training notebook)
    input_df['Sex'] = input_df['Sex'].map({'male': 1, 'female': 0})
    
    # One-hot encode 'Embarked' (drop_first=True, as per training notebook)
    input_df = pd.get_dummies(input_df, columns=['Embarked'], drop_first=True)

    # Ensure all columns expected by the model are present and in the correct order
    expected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    return input_df

# Make prediction when button is clicked
if st.sidebar.button('🔮 Predict Survival', use_container_width=True):
    processed_input = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
    
    try:
        prediction_proba = model.predict_proba(processed_input)[:, 1][0]
        prediction = (prediction_proba > 0.5).astype(int)

        st.subheader('Prediction Result:')
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.success(f"✅ Likely to Survive")
            else:
                st.error(f"❌ Likely to Not Survive")
        
        with col2:
            st.metric("Survival Probability", f"{prediction_proba:.2%}")
        
        # Show confidence
        confidence = max(prediction_proba, 1 - prediction_proba)
        st.info(f"Model confidence: {confidence:.2%}")

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")

# Instructions
st.write("\n---\n")
st.sidebar.write("**How to run this application locally:**")
st.sidebar.write("""
1. Ensure both `main.py` and `Titanic_train.csv` are in the same directory
2. Install required packages:
   ```
   pip install streamlit pandas scikit-learn
   ```
3. Run the command:
   ```
   streamlit run main.py
   ```
4. Your browser will automatically open the Streamlit application
""")
