import pandas as pd
from typing import Any, Generator, Tuple


class Association:
    def __init__(self, record, label, confidence=1):
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
    def __init__(self, identifier, is_reference, associations):
        self.identifier = identifier
        self.is_reference = is_reference
        self.associations = pd.DataFrame(
            [dict(association) for association in associations]
        )
        self.records = [association.record for association in associations]
        self.is_faulty = len(self.associations) == 0
        self.quality = {}
        self.quantity = {}


class NoisyLabelMatrix:
    def __init__(self, vectors):
        vectors_reference = [vector for vector in vectors if vector.is_reference]
        assert (
            len(vectors_reference) == 1
        ), "Only one vector should be the reference vector"
        self.vector_reference = vectors_reference[0]
        vectors = [vector for vector in vectors if not vector.is_reference]
        vectors.sort(
            key=lambda x: x.identifier
        )  # ensure that the vectors are sorted by their identifier
        self.vectors_noisy = vectors
        records = [vector.records for vector in self.vectors_noisy] + [
            self.vector_reference.records
        ]
        self.records = set([item for sublist in records for item in sublist])


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
