import plotly.graph_objects as go
import streamlit as st

# Initialize session state for two separate value lists
if "value_list_1" not in st.session_state:
    st.session_state.value_list_1 = []
if "value_list_2" not in st.session_state:
    st.session_state.value_list_2 = []

# Sidebar for first value input
st.sidebar.header("Add Values")
val1 = st.sidebar.number_input("Value for Series 1", key="val1", value=0.0)
if st.sidebar.button("Add to Series 1"):
    st.session_state.value_list_1.append(val1)

val2 = st.sidebar.number_input("Value for Series 2", key="val2", value=0.0)
if st.sidebar.button("Add to Series 2"):
    st.session_state.value_list_2.append(val2)

# Main area
st.title("Interactive Plot with Two Series")

# Show values
st.subheader("Current Series Values")
col1, col2 = st.columns(2)
with col1:
    st.write("Series 1:", st.session_state.value_list_1)
with col2:
    st.write("Series 2:", st.session_state.value_list_2)

# Plotting with Plotly
fig = go.Figure()

if st.session_state.value_list_1:
    fig.add_trace(
        go.Scatter(y=st.session_state.value_list_1, mode="lines+markers", name="Series 1")
    )

if st.session_state.value_list_2:
    fig.add_trace(
        go.Scatter(y=st.session_state.value_list_2, mode="lines+markers", name="Series 2")
    )

fig.update_layout(
    title="Plot of Entered Values", xaxis_title="Entry Index", yaxis_title="Value"
)

st.plotly_chart(fig, use_container_width=True)
