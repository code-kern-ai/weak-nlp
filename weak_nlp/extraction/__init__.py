from weak_nlp import base


class ExtractionAssociation(base.Association):
    """Record <> Label Association, e.g. for ground truths or heuristics

    Args:
        record (str): Identification for record
        label (str): Label name
        chunk_idx_start (_type_): Beginning of the chunk
        chunk_idx_end (_type_): End of the chunk
        confidence (Optional[float], optional): Confidence of the mapping. Defaults to 1.
    """

    def __init__(self, record, label, chunk_idx_start, chunk_idx_end, confidence=1):
        super().__init__(record, label, confidence)
        self.chunk_idx_start = chunk_idx_start
        self.chunk_idx_end = chunk_idx_end


class ExtractionNLM(base.NoisyLabelMatrix):
    """Collection of extraction source vectors that can be analyzed w.r.t.
    quality metrics (such as the confusion matrix, i.e., true positives etc.),
    quantity metrics (intersections and conflicts) or weakly supervisable labels.

    Args:
        vectors (List[SourceVector]): Containing the source entities for the matrix

    Raises:
        exceptions.MissingReferenceException: If this raises, you have set to many reference source vectors
    """
