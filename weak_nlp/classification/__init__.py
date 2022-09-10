from weak_nlp import base
import pandas as pd
from weak_nlp.shared import common_util, exceptions


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

    def quality_metrics(self) -> pd.DataFrame:
        if self.vector_reference is None:
            raise exceptions.MissingReferenceException(
                "Can't calculate the quality metrics without reference vector"
            )
        self._set_quality_metrics_inplace()

        statistics = []
        for vector_noisy in self.vectors_noisy:
            vector_stats = {"identifier": vector_noisy.identifier}
            for label_name in vector_noisy.quality.keys():
                vector_stats["label_name"] = label_name
                quality = vector_noisy.quality[label_name]

                vector_stats["true_positives"] = quality["true_positives"]
                vector_stats["false_positives"] = quality["false_positives"]
                vector_stats["false_negatives"] = quality["false_negatives"]
                statistics.append(vector_stats.copy())

        stats_df = pd.DataFrame(statistics)
        if len(stats_df) > 0:
            stats_df["precision"] = stats_df.apply(common_util.calc_precision, axis=1)
            stats_df["recall"] = stats_df.apply(common_util.calc_recall, axis=1)
        return stats_df

    def quantity_metrics(self) -> pd.DataFrame:
        self._set_quantity_metrics_inplace()

        statistics = []
        for vector_noisy in self.vectors_noisy:
            vector_stats = {"identifier": vector_noisy.identifier}
            for label_name in vector_noisy.quantity.keys():
                vector_stats["label_name"] = label_name

                quantity = vector_noisy.quantity[label_name]
                vector_stats["record_coverage"] = quantity["record_coverage"]
                vector_stats["source_conflicts"] = quantity["source_conflicts"]
                vector_stats["source_overlaps"] = quantity["source_overlaps"]
                statistics.append(vector_stats.copy())

        stats_df = pd.DataFrame(statistics)
        return stats_df
