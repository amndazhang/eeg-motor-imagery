from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

def create_riemannian_pipeline() -> Pipeline:
    return Pipeline([
        ('Covariances', Covariances(estimator='lwf')),
        ('TangentSpace', TangentSpace(metric='riemann')),
        ('Classifier', LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000))
    ])