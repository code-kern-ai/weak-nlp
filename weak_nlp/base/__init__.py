from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Generator, Tuple, List, Optional

from weak_nlp.shared import exceptions


def create_generator(instance: Any) -> Generator[Any, Tuple[str, Any], None]:
    for key in instance.__dict__:
        if type(getattr(instance, key)) == dict:
            yield key, dict(getattr(instance, key))
        else:
            yield key, getattr(instance, key)


def create_representational_string(instance: Any) -> str:
    representational_string = f"{instance.__module__}.{instance.__class__.__name__}("
    for var in vars(instance):
        val = repr(getattr(instance, var))
        representational_string += f"{var}={val},"
    return f"{representational_string[:-2]})"


def create_display_string(instance: Any) -> str:
    return str(dict(iter(instance)))


class Association(ABC):
    """Record <> Label Association, e.g. for ground truths or heuristics

    Args:
        record (str): Identification for record
        label (str): Label name
        confidence (Optional[float], optional): Confidence of the mapping. Defaults to 1.
    """

    def __init__(self, record: str, label: str, confidence: Optional[float] = 1):
        self.record = record
        self.label = label
        self.confidence = confidence

    def __repr__(self) -> str:
        return create_representational_string(self)

    def __str__(self) -> str:
        return create_display_string(self)

    def __iter__(self) -> Generator[Any, Tuple[str, Any], None]:
        return create_generator(self)

    def __getitem__(self, item) -> Any:
        return getattr(self, item)


class SourceVector:
    """Combines the created associations from one logical source.
    Additionally, it marks whether the respective source vector can be seen as a reference vector,
    such as a manually labeled source vector containing the *true* record <> label mappings.

    Args:
        identifier (str): Name of the source
        is_reference (bool): If set to True, this is seen as the ground truth
        associations (List[Association]): Actual mappings
    """

    def __init__(
        self, identifier: str, is_reference: bool, associations: List[Association]
    ):
        self.identifier = identifier
        self.is_reference = is_reference
        self.associations = pd.DataFrame(
            [dict(association) for association in associations]
        )
        self.records = [association.record for association in associations]
        self.is_empty = len(self.associations) == 0
        self.quality = {}
        self.quantity = {}


class NoisyLabelMatrix(ABC):
    """Collection of source vectors that can be analyzed w.r.t. quality metrics (such as the confusion matrix, i.e., true positives etc.),
    quantity metrics (intersections and conflicts) or weakly supervisable labels.

    Args:
        vectors (List[SourceVector]): Containing the source entities for the matrix

    Raises:
        exceptions.MissingReferenceException: If this raises, you have set to many reference source vectors
    """

    def __init__(self, vectors: List[SourceVector]):
        vectors_reference = [vector for vector in vectors if vector.is_reference]
        if len(vectors_reference) > 1:
            raise exceptions.MissingReferenceException(
                "Only one vector should be the reference vector"
            )
        if len(vectors_reference) == 1:
            self.vector_reference = vectors_reference[0]
        else:
            self.vector_reference = None
        vectors = [vector for vector in vectors if not vector.is_reference]
        vectors.sort(
            key=lambda x: x.identifier
        )  # ensure that the vectors are sorted by their identifier
        self.vectors_noisy = vectors
        records = [vector.records for vector in self.vectors_noisy]
        if self.vector_reference is not None:
            records += [self.vector_reference.records]
        self.records = set([item for sublist in records for item in sublist])

    @abstractmethod
    def _set_quality_metrics_inplace(self) -> None:
        """Calculate quality metrics true positives, false positives and false negatives inplace"""
        pass

    @abstractmethod
    def _set_quantity_metrics_inplace(self) -> None:
        """Calculate quantity metrics record coverage, total hits, source conflicts and source overlaps inplace"""
        pass

    @abstractmethod
    def quality_metrics(self) -> pd.DataFrame:
        """Retrieve calculates metrics as a dataframe

        Returns:
            pd.DataFrame: Containing the data per source and label
        """
        pass

    @abstractmethod
    def quantity_metrics(self) -> pd.DataFrame:
        """Retrieve calculates metrics as a dataframe

        Returns:
            pd.DataFrame: Containing the data per source and label
        """
        pass

    @abstractmethod
    def weakly_supervise(self, c: int, k: int) -> pd.Series:
        """Integrate existing noisy source vectors into one weakly supervised vector (as pandas series)

        Args:
            c (int): slope of the function
            k (int): what input should yield 0.5 probability?

        Returns:
            pd.Series: Containing the weakly supervised labels
        """
        pass
