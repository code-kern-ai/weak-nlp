from weak_nlp import SourceVector
from weak_nlp.classification import CNLM, ClassificationAssociation


def cnlm_from_df(df, lfs, target=None):
    source_vectors = []
    for lf in lfs:
        associations = []
        for idx, value in df[lf.__name__].dropna().to_dict().items():
            associations.append(ClassificationAssociation(f"r_{idx}", value))
        source_vectors.append(SourceVector(lf.__name__, False, associations))

    if target is not None:
        associations = []
        for idx, value in df[target].dropna().to_dict().items():
            associations.append(ClassificationAssociation(f"r_{idx}", value))
        source_vectors.append(SourceVector("manual", True, associations))
    return CNLM(source_vectors)
