import numpy as np


class DiscreteHMMTrainer:
    """
    Discrete Hidden Markov Model trained using Baum-Welch algorithm.
    """

    def __init__(self, n_states, n_observations, random_seed=None):
        self.N = n_states
        self.M = n_observations

        if random_seed is not None:
            np.random.seed(random_seed)

        self._initialize_parameters()

        # Storage for visualization / analysis
        self.history = {
            "A": [],
            "B": [],
            "pi": [],
            "log_likelihood": []
        }

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _initialize_parameters(self):

        self.A = np.random.rand(self.N, self.N)
        self.A /= self.A.sum(axis=1, keepdims=True)

        self.B = np.random.rand(self.N, self.M)
        self.B /= self.B.sum(axis=1, keepdims=True)

        self.pi = np.random.rand(self.N)
        self.pi /= self.pi.sum()

    # ============================================================
    # FORWARD (SCALED)
    # ============================================================

    def _forward(self, observations):

        T = len(observations)
        alpha = np.zeros((T, self.N))
        scales = np.zeros(T)

        alpha[0] = self.pi * self.B[:, observations[0]]
        scales[0] = alpha[0].sum()
        alpha[0] /= scales[0]

        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t - 1] * self.A[:, j]) \
                              * self.B[j, observations[t]]

            scales[t] = alpha[t].sum()
            alpha[t] /= scales[t]

        return alpha, scales

    # ============================================================
    # BACKWARD (SCALED)
    # ============================================================

    def _backward(self, observations, scales):

        T = len(observations)
        beta = np.zeros((T, self.N))

        beta[T - 1] = 1.0 / scales[T - 1]

        for t in reversed(range(T - 1)):
            for i in range(self.N):
                beta[t, i] = np.sum(
                    self.A[i] *
                    self.B[:, observations[t + 1]] *
                    beta[t + 1]
                )
            beta[t] /= scales[t]

        return beta

    # ============================================================
    # GAMMA & XI
    # ============================================================

    def _compute_gamma_xi(self, observations, alpha, beta):

        T = len(observations)
        gamma = np.zeros((T, self.N))
        xi = np.zeros((T - 1, self.N, self.N))

        for t in range(T - 1):
            denominator = np.sum(
                alpha[t][:, None] *
                self.A *
                self.B[:, observations[t + 1]] *
                beta[t + 1]
            )

            for i in range(self.N):
                numerator = alpha[t, i] * self.A[i] * \
                            self.B[:, observations[t + 1]] * \
                            beta[t + 1]

                xi[t, i] = numerator / denominator

            gamma[t] = xi[t].sum(axis=1)

        gamma[T - 1] = alpha[T - 1]

        return gamma, xi

    # ============================================================
    # RE-ESTIMATION
    # ============================================================

    def _reestimate(self, observations, gamma, xi):

        self.pi = gamma[0]

        for i in range(self.N):
            denominator = np.sum(gamma[:-1, i])
            for j in range(self.N):
                numerator = np.sum(xi[:, i, j])
                self.A[i, j] = numerator / denominator

        for j in range(self.N):
            denominator = np.sum(gamma[:, j])
            for k in range(self.M):
                mask = (observations == k)
                numerator = np.sum(gamma[mask, j])
                self.B[j, k] = numerator / denominator

    # ============================================================
    # LOG-LIKELIHOOD
    # ============================================================

    def _compute_log_likelihood(self, scales):
        return -np.sum(np.log(scales))

    # ============================================================
    # TRAINING LOOP (NOW TRACKS HISTORY)
    # ============================================================

    def train(self, observations, max_iter=100, tol=1e-4, verbose=True):

        observations = np.array(observations)

        # Reset history for fresh training
        self.history = {
            "A": [],
            "B": [],
            "pi": [],
            "log_likelihood": []
        }

        for iteration in range(max_iter):

            alpha, scales = self._forward(observations)
            beta = self._backward(observations, scales)
            gamma, xi = self._compute_gamma_xi(observations, alpha, beta)

            self._reestimate(observations, gamma, xi)

            log_likelihood = self._compute_log_likelihood(scales)

            # Store copies for visualization
            self.history["A"].append(self.A.copy())
            self.history["B"].append(self.B.copy())
            self.history["pi"].append(self.pi.copy())
            self.history["log_likelihood"].append(log_likelihood)

            if verbose:
                print(f"Iteration {iteration+1}: Log-Likelihood = {log_likelihood:.6f}")

            if iteration > 0:
                if abs(self.history["log_likelihood"][-1] -
                       self.history["log_likelihood"][-2]) < tol:
                    print("Converged.")
                    break

        return self.history
    

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def generate_synthetic_hmm_data(A, B, pi, length):
    """
    Generate hidden state sequence and observation sequence
    from a known HMM.
    """

    N = A.shape[0]
    states = np.zeros(length, dtype=int)
    observations = np.zeros(length, dtype=int)

    states[0] = np.random.choice(N, p=pi)
    observations[0] = np.random.choice(B.shape[1], p=B[states[0]])

    for t in range(1, length):
        states[t] = np.random.choice(N, p=A[states[t - 1]])
        observations[t] = np.random.choice(B.shape[1], p=B[states[t]])

    return states, observations


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    np.random.seed(0)

    # True HMM parameters
    true_A = np.array([[0.7, 0.3],
                       [0.4, 0.6]])

    true_B = np.array([[0.6, 0.3, 0.1],
                       [0.1, 0.4, 0.5]])

    true_pi = np.array([0.8, 0.2])

    # Generate synthetic sequence
    hidden_states, observations = generate_synthetic_hmm_data(
        true_A, true_B, true_pi, length=200
    )

    print("Generated Observation Sequence (first 20):")
    print(observations[:20])

    # Train HMM from random initialization
    hmm = DiscreteHMMTrainer(n_states=2, n_observations=3, random_seed=42)

    print("\nTraining HMM on synthetic data...\n")

    log_history = hmm.fit(observations, max_iters=100, tol=1e-6)

    print("\n============================")
    print("True vs Learned Parameters")
    print("============================")

    print("\nTrue A:\n", true_A)
    print("Learned A:\n", hmm.A)

    print("\nTrue B:\n", true_B)
    print("Learned B:\n", hmm.B)

    print("\nTrue pi:\n", true_pi)
    print("Learned pi:\n", hmm.pi)

    from diagram_generator import draw_hmm_transition_diagram

    states = ["S0", "S1"]
    draw_hmm_transition_diagram(states, hmm.A)