from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_csp_lda_pipeline(n_components: int = 4) -> Pipeline:
    """
    Constructs a spatial filtering (CSP) + feature scaling + LDA classification pipeline.
    """
    csp = CSP(n_components=n_components, log=True, norm_trace=False)
    scaler = StandardScaler()
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    return Pipeline([('CSP', csp), ('Scaler', scaler), ('LDA', lda)])