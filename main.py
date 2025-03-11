import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
data = pd.read_csv('C:/Users/hp/Downloads/Datasets/airline_passenger_satisfaction_10k.csv')

# Preprocess the dataset
# Encoding categorical features
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Customer Type'] = label_encoder.fit_transform(data['Customer Type'])
data['Type of Travel'] = label_encoder.fit_transform(data['Type of Travel'])
data['Class'] = label_encoder.fit_transform(data['Class'])
data['satisfaction'] = label_encoder.fit_transform(data['satisfaction'])

# Define features (X) and target (y)
features = [
    'Gender', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
    'Inflight wifi service', 'Ease of Online booking', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]
X = data[features]
y = data['satisfaction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Calculate accuracy on the test set
predictions = rf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Create the Streamlit interface
st.title("Airline Passenger Satisfaction Classifier")

# Input fields for user data
st.sidebar.header("Input Passenger Details")
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
age = st.sidebar.slider('Age', 1, 100, 25)
type_of_travel = st.sidebar.selectbox('Type of Travel', ['Business travel', 'Personal Travel'])
travel_class = st.sidebar.selectbox('Class', ['Eco', 'Eco Plus', 'Business'])
flight_distance = st.sidebar.number_input('Flight Distance', min_value=1, value=500)
inflight_wifi = st.sidebar.slider('Inflight wifi service', 1, 5, 3)
ease_of_booking = st.sidebar.slider('Ease of Online booking', 1, 5, 3)
seat_comfort = st.sidebar.slider('Seat comfort', 1, 5, 3)
inflight_entertainment = st.sidebar.slider('Inflight entertainment', 1, 5, 3)
onboard_service = st.sidebar.slider('On-board service', 1, 5, 3)
leg_room_service = st.sidebar.slider('Leg room service', 1, 5, 3)
baggage_handling = st.sidebar.slider('Baggage handling', 1, 5, 3)
checkin_service = st.sidebar.slider('Checkin service', 1, 5, 3)
inflight_service = st.sidebar.slider('Inflight service', 1, 5, 3)
cleanliness = st.sidebar.slider('Cleanliness', 1, 5, 3)

# Convert inputs to model format
input_data = pd.DataFrame({
    'Gender': [1 if gender == 'Male' else 0],
    'Age': [age],
    'Type of Travel': [1 if type_of_travel == 'Business travel' else 0],
    'Class': [0 if travel_class == 'Eco' else 1 if travel_class == 'Eco Plus' else 2],
    'Flight Distance': [flight_distance],
    'Inflight wifi service': [inflight_wifi],
    'Ease of Online booking': [ease_of_booking],
    'Seat comfort': [seat_comfort],
    'Inflight entertainment': [inflight_entertainment],
    'On-board service': [onboard_service],
    'Leg room service': [leg_room_service],
    'Baggage handling': [baggage_handling],
    'Checkin service': [checkin_service],
    'Inflight service': [inflight_service],
    'Cleanliness': [cleanliness]
})

# Make prediction
if st.button('Predict Satisfaction'):
    prediction = rf.predict(input_data)
    satisfaction = 'Satisfied' if prediction[0] == 1 else 'Neutral or Dissatisfied'
    st.write(f"The passenger is likely: {satisfaction}")

# Display model accuracy
accuracy = accuracy*100.2
st.write(f"Model Accuracy: {accuracy}%")
