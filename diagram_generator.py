"""
diagram_generator.py

Reusable visualization module for drawing a Hidden Markov Model
state transition diagram from a transition matrix.

Dependencies:
    - numpy
    - networkx
    - matplotlib

This module is designed to be easily integrated into:
    - Jupyter notebooks
    - Research scripts
    - Streamlit applications
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def draw_hmm_transition_diagram(
    states,
    A,
    threshold=0.01,
    figsize=(6, 6),
    node_color="#A0CBE2",
    edge_color="black",
    font_size=12,
    edge_font_size=10,
    title="HMM State Transition Diagram",
    return_fig=False,
):
    """
    Draws a state transition diagram for a Hidden Markov Model.

    Parameters
    ----------
    states : list of str
        List of state names. Length must match number of states in A.

    A : np.ndarray (N x N)
        Transition matrix where:
            A[i, j] = P(state_j | state_i)

    threshold : float, optional (default=0.01)
        Ignore transitions with probability below this value.

    figsize : tuple
        Size of matplotlib figure.

    node_color : str
        Color of state nodes.

    edge_color : str
        Color of transition edges.

    font_size : int
        Font size for state labels.

    edge_font_size : int
        Font size for probability labels.

    title : str
        Plot title.

    return_fig : bool
        If True, returns (fig, ax) for integration with Streamlit.

    Returns
    -------
    fig, ax (optional)
        Returned only if return_fig=True.
    """

    # --- Validation ---
    A = np.array(A)
    n_states = len(states)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Transition matrix A must be square.")

    if A.shape[0] != n_states:
        raise ValueError("Number of states must match size of A.")

    # --- Create Directed Graph ---
    G = nx.DiGraph()

    # Add nodes
    for state in states:
        G.add_node(state)

    # Add edges for probabilities above threshold
    for i in range(n_states):
        for j in range(n_states):
            prob = A[i, j]
            if prob >= threshold:
                G.add_edge(states[i], states[j], weight=prob)

    # --- Layout ---
    pos = nx.circular_layout(G)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_color,
        node_size=2000,
        ax=ax,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=font_size,
        font_weight="bold",
        ax=ax,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_color,
        arrows=True,
        arrowstyle="->",
        arrowsize=20,
        width=1.5,
        connectionstyle="arc3,rad=0.1",  # slight curve for clarity
        ax=ax,
    )

    # Edge labels (formatted probabilities)
    edge_labels = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in G.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=edge_font_size,
        ax=ax,
    )

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()

    if return_fig:
        return fig, ax
    else:
        plt.show()


# ---------------------------------------------------------
# Example usage (runs only if file executed directly)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example transition matrix
    states = ["S0", "S1", "S2"]

    A = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.6, 0.1],
        [0.05, 0.25, 0.7]
    ])

    draw_hmm_transition_diagram(states, A)