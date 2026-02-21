"""
app.py

Streamlit UI for training and visualizing a Discrete Hidden Markov Model
using Baum–Welch algorithm.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from baum_welch import DiscreteHMMTrainer
from diagram_generator import draw_hmm_transition_diagram


# -------------------------------------------------------
# Utility Functions
# -------------------------------------------------------

def random_stochastic_matrix(rows, cols):
    """Generate random row-stochastic matrix."""
    M = np.random.rand(rows, cols)
    return M / M.sum(axis=1, keepdims=True)


def generate_synthetic_sequence(A, B, pi, length):
    """
    Generate synthetic observation sequence from an HMM.
    """
    n_states = A.shape[0]
    n_obs = B.shape[1]

    states = []
    observations = []

    # Initial state
    state = np.random.choice(n_states, p=pi)
    states.append(state)

    obs = np.random.choice(n_obs, p=B[state])
    observations.append(obs)

    for _ in range(1, length):
        state = np.random.choice(n_states, p=A[state])
        states.append(state)

        obs = np.random.choice(n_obs, p=B[state])
        observations.append(obs)

    return np.array(observations)


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------

st.set_page_config(page_title="HMM Baum–Welch Visualizer", layout="wide")

st.title("📊 Hidden Markov Model – Baum–Welch Trainer")

st.sidebar.header("Model Configuration")

n_states = st.sidebar.slider("Number of States", 2, 6, 2)
n_observations = st.sidebar.slider("Number of Observation Symbols", 2, 6, 3)
sequence_length = st.sidebar.slider("Observation Sequence Length", 50, 1000, 200)
max_iter = st.sidebar.slider("Max Training Iterations", 10, 200, 50)
tol = st.sidebar.number_input("Convergence Tolerance", value=1e-4, format="%.6f")

run_button = st.sidebar.button("Train Model")


# -------------------------------------------------------
# Training Pipeline
# -------------------------------------------------------

if run_button:

    st.subheader("🔬 Generating Synthetic Data")

    # True parameters (used only to generate sequence)
    A_true = random_stochastic_matrix(n_states, n_states)
    B_true = random_stochastic_matrix(n_states, n_observations)
    pi_true = random_stochastic_matrix(1, n_states).flatten()

    observations = generate_synthetic_sequence(
        A_true, B_true, pi_true, sequence_length
    )

    st.success("Synthetic observation sequence generated.")

    # ---------------------------------------------------
    # Initialize Trainer
    # ---------------------------------------------------

    st.subheader("🚀 Training HMM (Baum–Welch)")

    trainer = DiscreteHMMTrainer(
        n_states=n_states,
        n_observations=n_observations
    )

    # Save initial parameters
    A_initial = trainer.A.copy()
    B_initial = trainer.B.copy()
    pi_initial = trainer.pi.copy()

    # Train model
    history = trainer.train(
        observations,
        max_iter=max_iter,
        tol=tol,
    )

    st.success("Training completed.")

    # Show final P(O | λ)
    st.subheader("📌 Final Likelihood")
    st.write("Final Log-Likelihood:", history["log_likelihood"][-1])

    # ---------------------------------------------------
    # Display Parameters
    # ---------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Initial Parameters")

        st.write("**Initial Transition Matrix (A):**")
        st.dataframe(A_initial)

        st.write("**Initial Emission Matrix (B):**")
        st.dataframe(B_initial)

        st.write("**Initial State Distribution (π):**")
        st.write(pi_initial)

    with col2:
        st.subheader("Final Parameters")

        st.write("**Final Transition Matrix (A):**")
        st.dataframe(trainer.A)

        st.write("**Final Emission Matrix (B):**")
        st.dataframe(trainer.B)

        st.write("**Final State Distribution (π):**")
        st.write(trainer.pi)

    # ---------------------------------------------------
    # Convergence Plot
    # ---------------------------------------------------

    st.subheader("📈 Log-Likelihood Convergence")

    log_likelihoods = history["log_likelihood"]

    fig_conv, ax_conv = plt.subplots()
    ax_conv.plot(log_likelihoods)
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Log-Likelihood")
    ax_conv.set_title("Baum–Welch Convergence")

    st.pyplot(fig_conv)

    # ---------------------------------------------------
    # Transition Diagram (Final)
    # ---------------------------------------------------

    st.subheader("🔁 Learned State Transition Diagram")

    state_names = [f"S{i}" for i in range(n_states)]

    fig, ax = draw_hmm_transition_diagram(
        state_names,
        trainer.A,
        return_fig=True
    )

    st.pyplot(fig)

    # ---------------------------------------------------
    # Iteration Slider Visualization
    # ---------------------------------------------------

    st.subheader("🎞 Transition Evolution Across Iterations")

    transition_history = history["A"]

    iteration_idx = st.slider(
        "Select Iteration",
        0,
        len(transition_history) - 1,
        len(transition_history) - 1
    )

    fig_iter, ax_iter = draw_hmm_transition_diagram(
        state_names,
        transition_history[iteration_idx],
        return_fig=True,
        title=f"Transition Matrix at Iteration {iteration_idx}"
    )

    st.pyplot(fig_iter)