from weak_nlp import base


class ClassificationAssociation(base.Association):
    """Record <> Label Association, e.g. for ground truths or heuristics

    Args:
        record (str): Identification for record
        label (str): Label name
        confidence (Optional[float], optional): Confidence of the mapping. Defaults to 1.
    """


class ClassificationNLM(base.NoisyLabelMatrix):
    """Collection of classification source vectors that can be analyzed w.r.t.
    quality metrics (such as the confusion matrix, i.e., true positives etc.),
    quantity metrics (intersections and conflicts) or weakly supervisable labels.

    Args:
        vectors (List[SourceVector]): Containing the source entities for the matrix

    Raises:
        exceptions.MissingReferenceException: If this raises, you have set to many reference source vectors
    """
