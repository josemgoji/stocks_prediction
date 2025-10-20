"""Construir estimadores tipo scikit-learn a partir de cla configuracion."""

from collections.abc import Mapping

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


ESTIMATOR_REGISTRY: dict[str, type[BaseEstimator]] = {
    "linear_regression": LinearRegression,
    "lasso": Lasso,
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "decision_tree": DecisionTreeRegressor,
    "svr": SVR,
    "knn": KNeighborsRegressor,
}


def create_estimator(name: str, params: Mapping[str, object] | None = None) -> BaseEstimator:
    """Devuelve un estimador configurado a partir del registro disponible."""
    try:
        estimator_cls = ESTIMATOR_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"El estimador '{name}' no está registrado. Opciones: {list(ESTIMATOR_REGISTRY)}"
        ) from exc

    kwargs = dict(params or {})
    return estimator_cls(**kwargs)
