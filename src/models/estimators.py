"""Factorías para construir estimadores scikit-learn a partir de configuración declarativa."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.base import BaseEstimator
from lightgbm import LGBMRegressor

EstimatorFactory = Callable[[Mapping[str, object] | None], BaseEstimator]


def _build_linear_regression(params: Mapping[str, object] | None) -> BaseEstimator:
    model = LinearRegression()
    if params:
        model.set_params(**params)
    return model


def _build_lasso(params: Mapping[str, object] | None) -> BaseEstimator:
    model = Lasso()
    if params:
        model.set_params(**params)
    return model


def _build_ridge(params: Mapping[str, object] | None) -> BaseEstimator:
    model = Ridge()
    if params:
        model.set_params(**params)
    return model


def _build_random_forest(params: Mapping[str, object] | None) -> BaseEstimator:
    model = RandomForestRegressor(random_state=params.get("random_state") if params else None)
    if params:
        model.set_params(**params)
    return model


def _build_gradient_boosting(params: Mapping[str, object] | None) -> BaseEstimator:
    model = GradientBoostingRegressor()
    if params:
        model.set_params(**params)
    return model


def _build_lightgbm(params: Mapping[str, object] | None) -> BaseEstimator:
    
    model = LGBMRegressor()
    if params:
        model.set_params(**params)
    return model


def _build_decision_tree(params: Mapping[str, object] | None) -> BaseEstimator:
    model = DecisionTreeRegressor(random_state=params.get("random_state") if params else None)
    if params:
        model.set_params(**params)
    return model


def _build_svr(params: Mapping[str, object] | None) -> BaseEstimator:
    model = SVR()
    if params:
        model.set_params(**params)
    return model


def _build_knn(params: Mapping[str, object] | None) -> BaseEstimator:
    model = KNeighborsRegressor()
    if params:
        model.set_params(**params)
    return model


ESTIMATOR_REGISTRY: dict[str, EstimatorFactory] = {
    "linear_regression": _build_linear_regression,
    "lasso": _build_lasso,
    "ridge": _build_ridge,
    "random_forest": _build_random_forest,
    "gradient_boosting": _build_gradient_boosting,
    "lightgbm": _build_lightgbm,
    "decision_tree": _build_decision_tree,
    "svr": _build_svr,
    "knn": _build_knn,
}


def create_estimator(name: str, params: Mapping[str, object] | None = None) -> BaseEstimator:
    """Devuelve un estimador configurado a partir del registro disponible."""
    try:
        factory = ESTIMATOR_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"El estimador '{name}' no está registrado. Opciones: {list(ESTIMATOR_REGISTRY)}"
        ) from exc
    return factory(params)
