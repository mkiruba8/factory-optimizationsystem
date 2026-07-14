import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    df = pd.read_csv("Nassau Candy Distributor.csv")

    df['Order Date'] = df['Order Date'].astype(str)
    df['Ship Date'] = df['Ship Date'].astype(str)

    # Explicit day-first format — removes the warning and parses faster
    df['Order Date'] = pd.to_datetime(
        df['Order Date'],
        format='%d-%m-%Y',
        errors='coerce'
    )

    df['Ship Date'] = pd.to_datetime(
        df['Ship Date'],
        format='%d-%m-%Y',
        errors='coerce'
    )

    df = df.dropna(subset=['Order Date', 'Ship Date'])

    df['Delivery Days'] = (df['Ship Date'] - df['Order Date']).dt.days

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
# Shipping Cost Category
# -----------------------------------

def shipping_category(cost):
    if cost < 500:
        return "Low"
    elif cost < 1000:
        return "Medium"
    else:
        return "High"

df["Shipping Category"] = df["Shipping Cost"].apply(shipping_category)
# ===============================
# Sidebar Filter
# ===============================

st.sidebar.header("Dashboard Filters")

selected_factory = st.sidebar.multiselect(
    "Select Factory",
    options=df["Factory"].unique(),
    default=df["Factory"].unique()
)

filtered_df = df[df["Factory"].isin(selected_factory)]

# Product Search
search_product = st.sidebar.text_input("🔍 Search Product")

if search_product:
    filtered_df = filtered_df[
        filtered_df["Product Name"].str.contains(
            search_product,
            case=False,
            na=False
        )
    ]
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

highest_cost = filtered_df["Shipping Cost"].max()

st.metric(
    "Highest Shipping Cost",
    f"${highest_cost:.2f}"
)
# -----------------------------------
# Factory-wise Analysis
# -----------------------------------
st.subheader("🏭 Factory-wise Shipping Cost")

factory_cost = filtered_df.groupby('Factory')['Shipping Cost'].mean()
fig, ax = plt.subplots()

factory_cost.plot(
    kind='bar',
    ax=ax
)

ax.set_ylabel("Average Shipping Cost")

st.pyplot(fig)

st.subheader("🥧 Factory Distribution")

factory_count = filtered_df["Factory"].value_counts()

fig, ax = plt.subplots()

ax.pie(
    factory_count,
    labels=factory_count.index,
    autopct="%1.1f%%",
    startangle=90
)

ax.axis("equal")

st.pyplot(fig)
st.subheader("📈 Distance vs Shipping Cost")

fig, ax = plt.subplots()

ax.plot(
    filtered_df["Distance KM"],
    filtered_df["Shipping Cost"],
    "o"
)

ax.set_xlabel("Distance (KM)")
ax.set_ylabel("Shipping Cost")

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
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction on the held-out test set
predictions = model.predict(X_test)

# -----------------------------------
# Model Evaluation
# -----------------------------------
st.subheader("📐 Model Evaluation")

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

eval_col1, eval_col2, eval_col3 = st.columns(3)

eval_col1.metric("MAE (Mean Absolute Error)", f"${mae:.2f}")
eval_col2.metric("RMSE (Root Mean Squared Error)", f"${rmse:.2f}")
eval_col3.metric("R² Score", f"{r2:.4f}")

st.caption(
    "MAE and RMSE are in the same unit as Shipping Cost (USD) — lower is better. "
    "R² Score shows the proportion of variance in Shipping Cost explained by the "
    "model, where a value closer to 1.0 indicates a better fit."
)

# Predicted vs Actual plot
st.markdown("**Predicted vs Actual Shipping Cost (Test Set)**")
fig, ax = plt.subplots()
ax.scatter(y_test, predictions, alpha=0.4, edgecolor="none")
lims = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
ax.plot(lims, lims, color="red", linestyle="--", linewidth=1, label="Perfect Prediction")
ax.set_xlabel("Actual Shipping Cost")
ax.set_ylabel("Predicted Shipping Cost")
ax.legend()
st.pyplot(fig)

# Residual plot
st.markdown("**Residuals (Actual − Predicted)**")
residuals = y_test - predictions
fig, ax = plt.subplots()
ax.scatter(predictions, residuals, alpha=0.4, edgecolor="none")
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Predicted Shipping Cost")
ax.set_ylabel("Residual")
st.pyplot(fig)

if mae < 30:
    st.success(f"Model Mean Absolute Error: {mae:.2f} — predictions are highly accurate.")
elif mae < 75:
    st.warning(f"Model Mean Absolute Error: {mae:.2f} — predictions are reasonably accurate.")
else:
    st.error(f"Model Mean Absolute Error: {mae:.2f} — consider more features or tuning.")

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

    input_df = pd.DataFrame(
        [[distance_input, delivery_input]],
        columns=['Distance KM', 'Delivery Days']
    )

    predicted_cost = model.predict(input_df)[0]

    # Recommendation Logic
    if distance_input <= 1000:
        if delivery_input <= 3:
            recommended_factory = "Factory A"
        else:
            recommended_factory = "Factory C"

    elif distance_input <= 2000:
        if delivery_input <= 5:
            recommended_factory = "Factory C"
        else:
            recommended_factory = "Factory B"

    else:
        recommended_factory = "Factory B"

    st.success(f"Predicted Shipping Cost: ${predicted_cost:.2f}")
    st.info(f"Recommended Factory: {recommended_factory}")

# -----------------------------------
# Factory Performance Summary
# -----------------------------------

st.subheader("🏭 Factory Performance Summary")

performance = filtered_df.groupby("Factory").agg({
    "Sales": "sum",
    "Gross Profit": "sum",
    "Shipping Cost": "mean",
    "Delivery Days": "mean"
}).round(2)

# -----------------------------------
# Factory Ranking
# -----------------------------------

st.subheader("🥇 Factory Ranking")

ranking = performance.sort_values(
    by="Gross Profit",
    ascending=False
)

ranking["Rank"] = range(1, len(ranking) + 1)

st.dataframe(ranking)
st.dataframe(performance)
# -----------------------------------
# Recommendation Summary
# -----------------------------------

st.subheader("💡 Recommendation Summary")

best_factory = filtered_df.groupby("Factory")["Shipping Cost"].mean().idxmin()

lowest_cost = filtered_df.groupby("Factory")["Shipping Cost"].mean().min()

st.success(f"🏆 Best Factory: {best_factory}")

st.write(f"Average Shipping Cost: ${lowest_cost:.2f}")

st.info(
    "Recommendation: Allocate more orders to this factory to reduce shipping costs and improve logistics performance."
)
# -------------------------------
# Top 5 Highest Shipping Cost
# -------------------------------

st.subheader("💰 Top 5 Highest Shipping Cost Orders")

top5 = filtered_df.nlargest(5, "Shipping Cost")

st.dataframe(
    top5[
        [
            "Product Name",
            "Factory",
            "Distance KM",
            "Shipping Cost"
        ]
    ]
)

st.subheader("📊 Shipping Cost Statistics")

st.write(filtered_df["Shipping Cost"].describe())
# -----------------------------------
# Show Full Dataset
# -----------------------------------
st.subheader("📋 Complete Processed Dataset")
st.dataframe(
    filtered_df[
        [
            "Product Name",
            "Factory",
            "Distance KM",
            "Shipping Cost",
            "Shipping Category",
            "Delivery Days"
        ]
    ]
)

# -----------------------------------
# Download CSV
# -----------------------------------
# -----------------------------------
# Download Filtered Dataset
# -----------------------------------

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇ Download Filtered Dataset",
    data=csv,
    file_name="Filtered_Shipping_Report.csv",
    mime="text/csv"
)
# -----------------------------------
# Dashboard Footer
# -----------------------------------

st.markdown("---")

st.markdown("""
### 👨‍💻 Developed By

**Kiruba M**
**M.Sc. Computer Science**
**Factory Reallocation & Shipping Optimization System**

📊 Built using **Python, Streamlit, Pandas, Scikit-learn, Matplotlib**
""")
