import ast
import math
import operator as op

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

# Supported operators and functions
SAFE_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

SAFE_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
}


def bisection_method(func, a, b):
    """
    Finds a root of a function using the bisection method.

    Args:
        func: The function for which to find the root.
        a: The lower bound of the initial interval.
        b: The upper bound of the initial interval.

    Returns:
        The approximate root of the function, or None if a root was not found.
    """
    f_a = eval_expr(func, a)
    f_b = eval_expr(func, b)
    if f_a * f_b >= 0:
        raise ValueError(
            "Function values at interval endpoints must have opposite signs."
        )

    midpoint = (a + b) / 2

    f_midpoint = eval_expr(func, midpoint)
    if f_a * f_midpoint > 0:
        a = midpoint
    else:
        b = midpoint

    return a, b


def NewtonRaphson(equation, derivative, x0):
    f_x = eval_expr(equation, x0)
    f_prime_x = eval_expr(derivative, x0)
    x0_new = x0 - f_x / f_prime_x
    return x0_new


def eval_expr(expr, x) -> float:
    """Safely evaluate a mathematical expression with variable x."""

    def _eval(node):
        if isinstance(node, ast.Constant):  # e.g., 2, 3.14
            return node.n
        elif isinstance(node, ast.BinOp):  # e.g., 2 + 3
            return SAFE_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):  # e.g., -1
            return SAFE_OPERATORS[type(node.op)](_eval(node.operand))
        elif isinstance(node, ast.Name):
            if node.id == "x":
                return x
            elif node.id in SAFE_FUNCTIONS:
                return SAFE_FUNCTIONS[node.id]
            else:
                raise ValueError(f"Unknown identifier: {node.id}")
        elif isinstance(node, ast.Call):  # e.g., sin(x)
            func = _eval(node.func)
            args = [_eval(arg) for arg in node.args]
            return func(*args)  # type: ignore
        else:
            raise TypeError(f"Unsupported type: {type(node)}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


def compute_derivative(exp):
    # Define the symbol
    x = sp.symbols("x")
    # Convert the string to a symbolic expression
    func = sp.sympify(exp)
    # Compute the derivative
    derivative = sp.diff(func, x)
    return str(derivative)


def reset_values():
    st.session_state.iterations_bisection = 0.0
    st.session_state.iterations_newton = 0.0
    st.session_state.vline1 = 2.0
    st.session_state.vline2 = 4.0
    st.session_state.x0Newton = 2.0


def framed_box(text):
    st.sidebar.markdown(
        f"""
    <div style="padding: 10px; border-radius: 5px; border: 1px solid #ddd; font-family: monospace;">
        {text}
    </div>
    """,
        unsafe_allow_html=True,
    )


def update_figure():
    x_vals = np.linspace(x_start, x_end, 300)
    y_vals = [eval_expr(equation, val) for val in x_vals]
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=f"f(x) = {equation}"))
    fig2.add_trace(
        go.Scatter(x=x_vals, y=y_vals, mode="lines", name=f"f(x) = {equation}")
    )
    # Add vertical lines
    fig.add_vline(
        x=st.session_state.vline1,
        line_dash="dash",
        line_color="red",
        annotation_text=f"x = {st.session_state.vline1}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=st.session_state.vline2,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"x = {st.session_state.vline2}",
        annotation_position="top right",
    )

    fig2.add_vline(
        x=st.session_state.x0Newton,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"x = {st.session_state.x0Newton}",
        annotation_position="top right",
    )

    x_vals_newton = np.linspace(st.session_state.vline1, st.session_state.vline2, 300)
    slope_newton = eval_expr(derivative_str, st.session_state.x0Newton)
    y0_newton = eval_expr(equation, st.session_state.x0Newton)
    f_prime_plot = y0_newton + slope_newton * (x_vals_newton - st.session_state.x0Newton)
    fig2.add_trace(
        go.Scatter(
            x=x_vals_newton,
            y=f_prime_plot,
            mode="lines",
            name=f"f'(x) = {derivative_str}",
            line_dash="dash",
            line_color="red",
        )
    )


# --- Streamlit App ---
# Initialize session state for x_start if not already set
if "vline1" not in st.session_state:
    reset_values()

# Sidebar input

# Equation input
st.sidebar.header("Plot an Equation")
equation = st.sidebar.text_input("Enter a function of x", value="sin(x)")
x_start = st.sidebar.number_input("Start of interval", value=0)
x_end = st.sidebar.number_input("End of interval", value=10)

derivative_str = compute_derivative(equation)
st.sidebar.write(f"The derivative of the function is:")
framed_box(derivative_str)


plot_equation = st.sidebar.button("Plot Equation")

# Vertical lines
st.sidebar.header("Bisection")

st.session_state.vline1 = st.sidebar.number_input(
    "Vertical Line 1 (x-position)",
    value=st.session_state.vline1,
    format="%.5f",
)

st.session_state.vline2 = st.sidebar.number_input(
    "Vertical Line 2 (x-position)",
    value=st.session_state.vline2,
    format="%.5f",
)

step_bisection = st.sidebar.button("Take step bisection")

st.sidebar.write("Bisection iterations:")
framed_box(st.session_state.iterations_bisection)

st.sidebar.header("Newton-Raphson")

st.session_state.x0Newton = st.sidebar.number_input(
    "Current x - Newton-Raphson",
    value=st.session_state.x0Newton,
    format="%.5f",
)
step_NewtonRaphson = st.sidebar.button("Take step Newton-Raphson")
st.sidebar.write("Newton-Raphson iterations:")
framed_box(st.session_state.iterations_newton)

reset_button = st.sidebar.button("Reset Button")


# Main area
st.title("Find Root Methods")

fig = go.Figure()
fig2 = go.Figure()

# Plot values
if plot_equation:
    update_figure()
# Plot safe user-defined equation
if step_bisection:
    try:
        st.session_state.vline1, st.session_state.vline2 = bisection_method(
            equation, st.session_state.vline1, st.session_state.vline2
        )
        st.session_state.iterations_bisection += 1
        update_figure()
    except Exception as e:
        st.error(f"Error: {e}")

if step_NewtonRaphson:
    try:
        st.session_state.iterations_newton += 1
        st.session_state.x0Newton = NewtonRaphson(
            equation, derivative_str, st.session_state.x0Newton
        )
        update_figure()
    except Exception as e:
        st.error(f"Error: {e}")

# Final plot layout
fig.update_layout(
    title=f"Bisection method for f(x) = {equation}", xaxis_title="x", yaxis_title="f(x)"
)
st.plotly_chart(fig, use_container_width=True)

fig2.update_layout(
    title=f"Newton-Raphson method of f(x) = {equation}",
    xaxis_title="x",
    yaxis_title="f(x)",
)
st.plotly_chart(fig2, use_container_width=True)


if reset_button:
    reset_values()
