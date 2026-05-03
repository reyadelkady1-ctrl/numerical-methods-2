import streamlit as st
import sympy as sp
import numpy as np
import time

st.set_page_config(page_title="Numerical Methods Pro", layout="wide")
st.title("📊 Numerical Methods Dashboard")

method = st.selectbox("Choose Method", [
    "Bisection",
    "False Position",
    "Newton-Raphson",
    "Secant",
    "Jacobi",
    "Gauss-Seidel"
])

tol = st.number_input("Tolerance", value=0.001)
max_iter = st.number_input("Max Iterations", value=50, min_value=1)

expr = None
a = None
b = None
x0 = None

if method in ["Bisection", "False Position", "Secant"]:
    expr = st.text_input("Function f(x)", "x**3 - x - 2")
    a = st.number_input("a", value=1.0)
    b = st.number_input("b", value=2.0)

elif method == "Newton-Raphson":
    expr = st.text_input("Function f(x)", "x**3 - x - 2")
    x0 = st.number_input("Initial Guess x0", value=1.5)

if method in ["Jacobi", "Gauss-Seidel"]:
    eq_input = st.text_area("Enter equations", "4*x + y = 1\nx + 3*y = 2")

def get_func(expr):
    x = sp.symbols('x')
    return sp.lambdify(x, sp.sympify(expr), "numpy")

def get_dfunc(expr):
    x = sp.symbols('x')
    return sp.lambdify(x, sp.diff(sp.sympify(expr), x), "numpy")

placeholder = st.empty()

# ================= BISECTION =================
def bisection(f, a, b):
    if f(a) * f(b) > 0:
        st.error("Invalid interval")
        return

    for i in range(int(max_iter)):
        c = (a + b) / 2

        placeholder.markdown(f"""
### 🔁 Iteration {i+1}
a = {a}  
b = {b}  
c = {c}  
f(c) = {f(c)}
        """)

        time.sleep(0.5)

        if abs(f(c)) < tol:
            break

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    st.success(f"Root ≈ {c}")

# ================= FALSE POSITION =================
def false_position(f, a, b):
    if f(a) * f(b) > 0:
        st.error("Invalid interval")
        return

    for i in range(int(max_iter)):
        if f(b) - f(a) == 0:
            st.error("Division by zero")
            return

        c = (a*f(b) - b*f(a)) / (f(b) - f(a))

        placeholder.markdown(f"""
### 🔁 Iteration {i+1}
a = {a}  
b = {b}  
c = {c}  
f(c) = {f(c)}
        """)

        time.sleep(0.5)

        if abs(f(c)) < tol:
            break

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    st.success(f"Root ≈ {c}")

# ================= NEWTON =================
def newton(f, df, x):
    for i in range(int(max_iter)):
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            st.error("Derivative is zero")
            return

        x_new = x - fx / dfx

        placeholder.markdown(f"""
### 🔁 Iteration {i+1}
x = {x}  
f(x) = {fx}  
f'(x) = {dfx}  
x_new = {x_new}
        """)

        time.sleep(0.5)

        if abs(fx) < tol:
            break

        x = x_new

    st.success(f"Root ≈ {x}")

# ================= SECANT =================
def secant(f, x0, x1):
    for i in range(int(max_iter)):
        fx0 = f(x0)
        fx1 = f(x1)

        if fx1 - fx0 == 0:
            st.error("Division by zero")
            return

        x2 = x1 - fx1*(x1-x0)/(fx1-fx0)

        placeholder.markdown(f"""
### 🔁 Iteration {i+1}
x0 = {x0}  
x1 = {x1}  
x2 = {x2}
        """)

        time.sleep(0.5)

        if abs(fx1) < tol:
            break

        x0, x1 = x1, x2

    st.success(f"Root ≈ {x2}")

# ================= JACOBI =================
def jacobi(A, B):
    n = len(B)
    x = np.zeros(n)

    for i in range(int(max_iter)):
        x_new = np.zeros(n)

        for j in range(n):
            s = sum(A[j][k]*x[k] for k in range(n) if k != j)
            x_new[j] = (B[j] - s) / A[j][j]

        placeholder.markdown(f"""
### 🔁 Jacobi Iteration {i+1}
{x_new}
        """)

        time.sleep(0.5)

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    st.success(f"Solution ≈ {x}")

# ================= GAUSS-SEIDEL =================
def gauss(A, B):
    n = len(B)
    x = np.zeros(n)

    for i in range(int(max_iter)):
        for j in range(n):
            s1 = sum(A[j][k]*x[k] for k in range(j))
            s2 = sum(A[j][k]*x[k] for k in range(j+1, n))
            x[j] = (B[j] - s1 - s2) / A[j][j]

        placeholder.markdown(f"""
### 🔁 Iteration {i+1}
{x}
        """)

        time.sleep(0.5)

        if np.linalg.norm(A @ x - B) < tol:
            break

    st.success(f"Solution ≈ {x}")

# ================= RUN =================
if st.button("Run"):

    try:

        # ================= SYSTEM METHODS =================
        if method in ["Jacobi", "Gauss-Seidel"]:

            if not eq_input:
                st.error("Enter equations first")
                st.stop()

            eqs = eq_input.strip().split("\n")

            exprs = []
            vars_set = set()

            for eq in eqs:
                l, r = eq.split("=")
                l = sp.sympify(l)
                r = sp.sympify(r)
                exprs.append(l - r)
                vars_set.update(l.free_symbols)

            vars_list = sorted(list(vars_set), key=lambda x: x.name)

            A, B = [], []

            for e in exprs:
                A.append([e.coeff(v) for v in vars_list])
                B.append(-e.subs({v: 0 for v in vars_list}))

            A = np.array(A, float)
            B = np.array(B, float)

            if method == "Jacobi":
                jacobi(A, B)
            else:
                gauss(A, B)

        # ================= ROOT METHODS =================
        else:

            if not expr:
                st.error("Function is empty")
                st.stop()

            f = get_func(expr)

            if method == "Newton-Raphson":
                df = get_dfunc(expr)

            if method == "Bisection":
                bisection(f, a, b)

            elif method == "False Position":
                false_position(f, a, b)

            elif method == "Newton-Raphson":
                newton(f, df, x0)

            elif method == "Secant":
                secant(f, a, b)

    except Exception as e:
        st.error(str(e))