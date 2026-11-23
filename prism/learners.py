"""Standard regression portfolio shared across predictor nodes."""
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_estimators():
    out = {
        "ridge_0.1": Ridge(alpha=0.1),
        "ridge_1": Ridge(alpha=1.0),
        "kernel": KernelRidge(alpha=0.03, kernel="rbf"),
        "trees": ExtraTreesRegressor(n_estimators=48, random_state=71001, n_jobs=1),
    }
    for depth in (2, 3):
        out[f"boost_{depth}"] = GradientBoostingRegressor(
            n_estimators=200, max_depth=depth, learning_rate=0.05, loss="huber", random_state=71001
        )
    for name, kernel in (("matern", Matern(length_scale=1.0, nu=2.5)), ("rbf", RBF(length_scale=1.0))):
        out[f"gp_{name}"] = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * kernel + WhiteKernel(1e-3, (1e-7, 0.5)),
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=71001,
        )
    out["mlp_64x3"] = TransformedTargetRegressor(
        regressor=MLPRegressor(
            hidden_layer_sizes=(64, 64, 64),
            solver="adam",
            learning_rate_init=0.01,
            alpha=1e-4,
            max_iter=1000,
            n_iter_no_change=30,
            random_state=71001,
        ),
        transformer=StandardScaler(),
    )
    return {k: make_pipeline(StandardScaler(), v) for k, v in out.items()}
