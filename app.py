import streamlit as st
import sympy as sp
import numpy as np
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Numerical Methods Pro", layout="wide")

st.title("📊 Numerical Methods Dashboard")

st.sidebar.header("Control Panel")

method = st.sidebar.selectbox("Choose Method", [
    "Bisection",
    "False Position",
    "Newton-Raphson",
    "Secant",
    "Jacobi",
    "Gauss-Seidel",
    "Compare All"
])

tol = st.sidebar.number_input("Tolerance", value=0.001)

expr = st.text_input("Function f(x)", "x**3 - x - 2")

a = st.number_input("a", value=1.0)
b = st.number_input("b", value=2.0)
x0 = st.number_input("x0", value=1.5)
x1 = st.number_input("x1", value=2.0)

n = 2
A = np.array([[4, 1],
              [1, 3]], dtype=float)
B = np.array([1, 2], dtype=float)

def f(expr):
    x = sp.symbols('x')
    return sp.lambdify(x, sp.sympify(expr), "numpy")

def df(expr):
    x = sp.symbols('x')
    return sp.lambdify(x, sp.diff(sp.sympify(expr), x), "numpy")

func = f(expr)
dfunc = df(expr)

def bisection():
    A_, B_ = a, b
    steps = []
    start = time.time()

    for i in range(50):
        c = (A_ + B_) / 2
        steps.append(c)

        if abs(func(c)) < tol:
            return c, i+1, time.time()-start, steps

        if func(A_) * func(c) < 0:
            B_ = c
        else:
            A_ = c

    return c, i+1, time.time()-start, steps


def false_position():
    A_, B_ = a, b
    steps = []
    start = time.time()

    for i in range(50):
        c = (A_ * func(B_) - B_ * func(A_)) / (func(B_) - func(A_))
        steps.append(c)

        if abs(func(c)) < tol:
            return c, i+1, time.time()-start, steps

        if func(A_) * func(c) < 0:
            B_ = c
        else:
            A_ = c

    return c, i+1, time.time()-start, steps


def newton():
    x = x0
    steps = []
    start = time.time()

    for i in range(50):
        x = x - func(x) / dfunc(x)
        steps.append(x)

        if abs(func(x)) < tol:
            return x, i+1, time.time()-start, steps

    return x, i+1, time.time()-start, steps


def secant():
    x_a, x_b = x0, x1
    steps = []
    start = time.time()

    for i in range(50):
        x_c = x_b - func(x_b) * (x_b - x_a) / (func(x_b) - func(x_a))
        steps.append(x_c)

        if abs(func(x_c)) < tol:
            return x_c, i+1, time.time()-start, steps

        x_a, x_b = x_b, x_c

    return x_c, i+1, time.time()-start, steps


def jacobi():
    x = np.zeros(n)
    start = time.time()

    for i in range(50):
        x_new = np.zeros(n)
        for j in range(n):
            s = sum(A[j][k] * x[k] for k in range(n) if k != j)
            x_new[j] = (B[j] - s) / A[j][j]

        if np.linalg.norm(x_new - x) < tol:
            return x_new, i+1, time.time()-start, [x_new]

        x = x_new

    return x, i+1, time.time()-start, [x]


def gauss_seidel():
    x = np.zeros(n)
    start = time.time()

    for i in range(50):
        for j in range(n):
            s1 = sum(A[j][k] * x[k] for k in range(j))
            s2 = sum(A[j][k] * x[k] for k in range(j+1, n))
            x[j] = (B[j] - s1 - s2) / A[j][j]

        if np.linalg.norm(A @ x - B) < tol:
            return x, i+1, time.time()-start, [x.copy()]

    return x, i+1, time.time()-start, [x]


def run(method_name):
    if method_name == "Bisection":
        return bisection()
    elif method_name == "False Position":
        return false_position()
    elif method_name == "Newton-Raphson":
        return newton()
    elif method_name == "Secant":
        return secant()
    elif method_name == "Jacobi":
        return jacobi()
    elif method_name == "Gauss-Seidel":
        return gauss_seidel()


if method != "Compare All":

    if st.button("Run"):

        res, it, t, steps = run(method)

        col1, col2, col3 = st.columns(3)

        col1.metric("Result", res)
        col2.metric("Iterations", it)
        col3.metric("Time", round(t, 6))

        st.subheader("🎬 Step Animation")

        placeholder = st.empty()

        for i, val in enumerate(steps):
            placeholder.metric(f"Iteration {i+1}", val)
            time.sleep(0.3)

        x_vals = np.linspace(-10, 10, 400)
        y_vals = func(x_vals)

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals)
        ax.axhline(0)

        try:
            ax.scatter(res, func(res))
        except:
            pass

        st.pyplot(fig)

else:

    if st.button("Compare"):

        methods = [
            "Bisection",
            "False Position",
            "Newton-Raphson",
            "Secant",
            "Jacobi",
            "Gauss-Seidel"
        ]

        names = []
        iters = []
        times = []

        for m in methods:
            r, i, t, _ = run(m)
            names.append(m)
            iters.append(i)
            times.append(t)

        best = names[np.argmin(times)]

        st.table({
            "Method": names,
            "Iterations": iters,
            "Time": times
        })

        st.bar_chart({
            "Iterations": dict(zip(names, iters)),
            "Time": dict(zip(names, times))
        })

        st.success(f"🏆 Best Method: {best}")