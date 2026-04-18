import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Rock vs Mine Prediction",
    page_icon="🪨",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv("data/sonar_data.csv", header=None)


@st.cache_resource
def train_model(dataframe):
    X = dataframe.drop(columns=[60])
    y = dataframe[60]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    return model, scaler, train_acc, test_acc, X


data = load_data()
model, scaler, train_acc, test_acc, X = train_model(data)

st.title("Rock vs Mine Prediction")
st.markdown(
    """
    Predict whether a sonar signal represents a **Rock** or a **Mine**
    using a trained Logistic Regression model.
    """
)

with st.expander("Model Information"):
    st.write("**Algorithm:** Logistic Regression")
    st.write(f"**Training Accuracy:** {train_acc:.3f}")
    st.write(f"**Testing Accuracy:** {test_acc:.3f}")
    st.write("**Input Features:** 60 sonar signal values")

st.divider()

st.subheader("Feature Input")

sample_index = st.selectbox(
    "Select a sample row to autofill",
    ["None"] + [str(i) for i in range(min(10, len(X)))]
)

if sample_index != "None":
    default_values = X.iloc[int(sample_index)].tolist()
else:
    default_values = [0.0] * 60

st.write("Adjust the 60 feature values below:")

input_values = []

for i in range(0, 60, 5):
    cols = st.columns(5)

    for j in range(5):
        idx = i + j
        with cols[j]:
            value = st.slider(
                f"F{idx + 1}",
                min_value=0.0,
                max_value=1.0,
                value=float(default_values[idx]),
                step=0.01
            )
            input_values.append(value)

st.divider()

if st.button("Predict"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    rock_prob = probabilities[list(model.classes_).index("R")]
    mine_prob = probabilities[list(model.classes_).index("M")]

    st.subheader("Prediction Result")

    result_col, prob_col = st.columns(2)

    with result_col:
        if prediction == "R":
            st.success("Prediction: Rock")
        else:
            st.error("Prediction: Mine")

    with prob_col:
        st.write(f"**Rock Probability:** {rock_prob:.2%}")
        st.write(f"**Mine Probability:** {mine_prob:.2%}")

st.divider()
st.caption("Built with Streamlit, Pandas, NumPy, and Scikit-learn.")
