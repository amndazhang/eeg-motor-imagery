import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

def create_csp_lda_pipeline(n_components: int = 4) -> Pipeline:
    """
    Constructs a spatial filtering (CSP) + feature scaling + LDA classification pipeline.
    """
    csp = CSP(n_components=n_components, log=True, norm_trace=False)
    scaler = StandardScaler()
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    return Pipeline([('CSP', csp), ('Scaler', scaler), ('LDA', lda)])

def create_riemannian_pipeline() -> Pipeline:
    """
    Constructs a Covariance -> Tangent Space Mapping -> Logistic Regression pipeline.
    Provides robust cross-subject classification with native probability estimation.
    """
    cov = Covariances(estimator='lwf')
    tangent = TangentSpace(metric='riemann')
    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    
    return Pipeline([
        ('Covariances', cov),
        ('TangentSpace', tangent),
        ('Classifier', clf)
    ])