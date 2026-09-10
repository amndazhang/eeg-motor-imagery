import numpy as np
import scipy.signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

class FilterBankTangentSpace(BaseEstimator, TransformerMixin):
    """
    Decomposes multi-channel EEG signals into multiple narrow frequency bands,
    computes Ledoit-Wolf covariance matrices per band, and projects them 
    into concatenated Riemannian Tangent Spaces.
    """
    def __init__(self, sfreq: float = 160.0, bands: list = None):
        self.sfreq = sfreq
        self.bands = bands or [
            (8, 12),   # Low mu
            (12, 16),  # High mu / Low beta
            (16, 20),  # Mid beta
            (20, 24),  # High beta
            (24, 28),  # Upper beta
            (28, 32)   # High beta boundary
        ]
        self.cov_estimators = [Covariances(estimator='lwf') for _ in self.bands]
        self.tangent_spaces = [TangentSpace(metric='riemann') for _ in self.bands]

    def _filter_band(self, X: np.ndarray, l_freq: float, h_freq: float) -> np.ndarray:
        nyq = 0.5 * self.sfreq
        b, a = scipy.signal.butter(4, [l_freq / nyq, h_freq / nyq], btype='band')
        return scipy.signal.filtfilt(b, a, X, axis=-1)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        for idx, (l_freq, h_freq) in enumerate(self.bands):
            X_band = self._filter_band(X, l_freq, h_freq)
            covs = self.cov_estimators[idx].fit_transform(X_band)
            self.tangent_spaces[idx].fit(covs)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        tangent_features = []
        for idx, (l_freq, h_freq) in enumerate(self.bands):
            X_band = self._filter_band(X, l_freq, h_freq)
            covs = self.cov_estimators[idx].transform(X_band)
            tangents = self.tangent_spaces[idx].transform(covs)
            tangent_features.append(tangents)
        return np.concatenate(tangent_features, axis=1)

def create_filterbank_riemann_pipeline(n_features_to_select: int = 120) -> Pipeline:
    """
    Pipeline combining Filter Bank Tangent mapping, Mutual Information feature 
    selection, and L2-regularized Logistic Regression.
    """
    return Pipeline([
        ('FilterBank', FilterBankTangentSpace()),
        ('FeatureSelect', SelectKBest(score_func=mutual_info_classif, k=n_features_to_select)),
        ('Classifier', LogisticRegression(C=0.15, solver='lbfgs', max_iter=1000))
    ])