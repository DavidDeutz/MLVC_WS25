# ---------- epsilon_svr.py ----------
import cvxopt
import numpy as np

########### TO-DO ###########
# Implement the methods marked with "BEGINNING OF YOUR CODE" and "END OF YOUR CODE":
# fit(), predict()
# Do not change the function signatures
# Do not change any other code
#############################

class EpsilonSVR:
    """
    ε-Support Vector Regression (dual form).

    This implementation works only with sklearn-compatible kernels.
    The kernel must be a callable with signature
        K(X, Y) -> ndarray of shape (len(X), len(Y)),
    e.g. an instance from sklearn.gaussian_process.kernels (RBF, Matern, etc.).

    Dual optimization problem
    -------------------------
    Given training data {(x_i, y_i)}_{i=1..n}, we solve for alpha, alpha* ∈ R^n:

        minimize   1/2 (alpha - alpha*)^T K (alpha - alpha*) + epsilon 1^T (alpha + alpha*) - y^T (alpha - alpha*)
        subject to 0 ≤ alpha_i ≤ C,
                   0 ≤ alpha*_i ≤ C,
                   1^T (alpha - alpha*) = 0.

    Here K is the kernel Gram matrix. The solution defines coefficients
    (alpha - alpha*) that weight support vectors in the prediction.

    Prediction
    ----------
    For a test point x,
        f(x) = Σ_i (alpha_i - alpha*_i) K(x_i, x) + b.

    Notes
    -----
    * C > 0 controls the regularization strength (penalty for large alpha, alpha*).
    * epsilon ≥ 0 defines the “epsilon-insensitive” zone around targets y_i where
      deviations incur no loss.
    * Input normalization (scaling by max norm) can be enabled for
      numerical stability.
    """

    def __init__(self, C=1.0, epsilon=0.1, kernel=None, normalize=True):
        if kernel is None:
            raise ValueError("Provide an sklearn-compatible kernel instance (callable K(X, Y)).")
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.__sk_kernel = kernel
        self.__normalize = bool(normalize)

        # Learned params
        self.__a = None             # a
        self.__a_star = None        # a*
        self.__coef = None          # (a - a*)
        self.__bias = 0.0
        self.__training_X = None    # numpy, scaled if normalize=True
        self.__norm = 1.0
        self.__support_mask = None  # boolean mask over training samples

    # ---- Sklearn-kernel bridge ----
    def _kernel(self, X1_np, X2_np):
        return self.__sk_kernel(X1_np, X2_np)

    def fit(self, X, y):
        """
        Fit ε-SVR in the dual form using quadratic programming.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input vectors.
        y : array-like of shape (n_samples,)
            Training target values.

        Task
        ----
        * Optionally normalize X for stability.
        * Build the kernel Gram matrix K(X, X).
        * Formulate the dual quadratic program in variables z = [alpha; alpha*].
        * Solve with a QP solver (e.g. cvxopt).
        * Extract alpha, alpha*, coefficients (alpha - alpha*), and identify support vectors.
        * Compute bias b from KKT conditions using near-margin samples.

        Returns
        -------
        self : EpsilonSVR
            Fitted model with dual variables and bias stored.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, _ = X.shape

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        raise NotImplementedError("Provide your solution here")
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        
        return self

    def _k_train_test(self, Xtest_scaled):
        return self._kernel(self.__training_X, Xtest_scaled)  # (n_train, n_test)

    def predict(self, X):
        """
        Predict regression outputs for new data using the dual form.

        For each test point x, compute:
            f(x) = Σ_i (alpha_i - alpha*_i) K(x_i, x) + b,
        where the sum runs over support vectors.

        Parameters
        ----------
        X : array-like of shape (m, n_features)
            Test input vectors.

        Returns
        -------
        y_pred : np.ndarray of shape (m,)
            Predicted target values.
        """

        if self.__coef is None:
            raise RuntimeError("Model is not fit yet.")

        X = np.asarray(X, dtype=np.float64)

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        raise NotImplementedError("Provide your solution here")
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return y_pred
    

