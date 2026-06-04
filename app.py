import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Factory Reallocation & Shipping Optimization",
    layout="wide"
)

st.title("📦 Factory Reallocation & Shipping Optimization System")
st.write("Nassau Candy Distributor Recommendation Dashboard")

# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():

    # Read CSV
    df = pd.read_csv("Nassau Candy Distributor.csv")

    # Convert to string first
    df['Order Date'] = df['Order Date'].astype(str)
    df['Ship Date'] = df['Ship Date'].astype(str)

    # Convert columns to datetime
    df['Order Date'] = pd.to_datetime(
        df['Order Date'],
        errors='coerce'
    )

    df['Ship Date'] = pd.to_datetime(
        df['Ship Date'],
        errors='coerce'
    )

    # Remove invalid rows
    df = df.dropna(subset=['Order Date', 'Ship Date'])

    # Delivery Days
    df['Delivery Days'] = (
        df['Ship Date'] - df['Order Date']
    ).dt.days

    return df

df = load_data()

# ==================================
# Executive Dashboard
# ==================================

st.subheader("📊 Executive Dashboard")

total_sales = df['Sales'].sum()
total_profit = df['Gross Profit'].sum()
total_orders = len(df)
avg_delivery = df['Delivery Days'].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
col2.metric("📈 Total Profit", f"${total_profit:,.2f}")
col3.metric("📦 Total Orders", total_orders)
col4.metric("🚚 Avg Delivery Days", round(avg_delivery, 2))

# Top Product Card

top_product = (
    df.groupby('Product Name')['Sales']
    .sum()
    .idxmax()
)

st.info(
    f"🏆 Best Selling Product: {top_product}"
)

profit_margin = (
    total_profit / total_sales) * 100

st.metric(
    "Profit Margin %",
    f"{profit_margin:.2f}%"
)

st.subheader("💡 Business Insights")

if profit_margin < 10:
    st.warning(
        "Profit margin is low. Review shipping costs."
    )

if avg_delivery > 5:
    st.warning(
        "Average delivery time is high."
    )
else:
    st.success(
        "Delivery performance is good."
    )

# -----------------------------------
# Show Dataset
# -----------------------------------

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -----------------------------------
# Sample Factory Coordinates
# -----------------------------------
factory_locations = {
    "Factory A": (40.7128, -74.0060),   # New York
    "Factory B": (34.0522, -118.2437),  # Los Angeles
    "Factory C": (41.8781, -87.6298)    # Chicago
}

# -----------------------------------
# Generate Random Customer Coordinates
# -----------------------------------
np.random.seed(42)

df['Customer Lat'] = np.random.uniform(25, 45, len(df))
df['Customer Lon'] = np.random.uniform(-120, -70, len(df))

# Randomly assign factory
df['Factory'] = np.random.choice(
    list(factory_locations.keys()),
    len(df)
)

# -----------------------------------
# Distance Calculation
# -----------------------------------
distances = []

for index, row in df.iterrows():

    customer_location = (
        row['Customer Lat'],
        row['Customer Lon']
    )

    factory_location = factory_locations[row['Factory']]

    distance = geodesic(
        factory_location,
        customer_location
    ).km

    distances.append(distance)

df['Distance KM'] = distances

# -----------------------------------
# Simulated Shipping Cost
# -----------------------------------
np.random.seed(100)

df['Shipping Cost'] = (
    df['Distance KM'] * 0.5
    + np.random.randint(20, 100, len(df))
)

# -----------------------------------
# Dashboard Metrics
# -----------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Orders",
    len(df)
)

col2.metric(
    "Average Shipping Cost",
    f"${df['Shipping Cost'].mean():.2f}"
)

col3.metric(
    "Average Delivery Days",
    round(df['Delivery Days'].mean(), 2)
)

# -----------------------------------
# Factory-wise Analysis
# -----------------------------------
st.subheader("🏭 Factory-wise Shipping Cost")

factory_cost = df.groupby('Factory')['Shipping Cost'].mean()

fig, ax = plt.subplots()

factory_cost.plot(
    kind='bar',
    ax=ax
)

ax.set_ylabel("Average Shipping Cost")

st.pyplot(fig)

# -----------------------------------
# Machine Learning Model
# -----------------------------------
st.subheader("🤖 Shipping Cost Prediction")

# Features
X = df[['Distance KM', 'Delivery Days']]

# Target
y = df['Shipping Cost']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
mae = mean_absolute_error(y_test, predictions)

st.success(f"Model Mean Absolute Error: {mae:.2f}")

# -----------------------------------
# Recommendation System
# -----------------------------------
st.subheader("✅ Factory Recommendation System")

distance_input = st.number_input(
    "Enter Distance (KM)",
    min_value=1.0,
    value=500.0
)

delivery_input = st.number_input(
    "Enter Delivery Days",
    min_value=1,
    value=5
)

if st.button("Predict Shipping Cost"):

    predicted_cost = model.predict(
        [[distance_input, delivery_input]]
    )[0]

    # Recommendation Logic
    if distance_input < 1000:
        recommended_factory = "Factory A"
    elif distance_input < 2000:
        recommended_factory = "Factory C"
    else:
        recommended_factory = "Factory B"

    st.success(
        f"Predicted Shipping Cost: ${predicted_cost:.2f}"
    )

    st.info(
        f"Recommended Factory: {recommended_factory}"
    )

# -----------------------------------
# Show Full Dataset
# -----------------------------------
st.subheader("📋 Complete Processed Dataset")

st.dataframe(df)

# -----------------------------------
# Download CSV
# -----------------------------------
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇ Download Processed Dataset",
    data=csv,
    file_name='processed_shipping_data.csv',
    mime='text/csv'
)
