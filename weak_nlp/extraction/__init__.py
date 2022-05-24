import weak_nlp
import numpy as np
import pandas as pd
from weak_nlp.shared import common_util, exceptions
from weak_nlp.extraction import util


class ExtractionAssociation(weak_nlp.Association):
    def __init__(self, record, label, chunk_idx_start, chunk_idx_end, confidence=1):
        super().__init__(record, label, confidence)
        self.chunk_idx_start = chunk_idx_start
        self.chunk_idx_end = chunk_idx_end


class ENLM(weak_nlp.NoisyLabelMatrix):
    def __init__(self, vectors):
        super().__init__(vectors)

    def _set_quality_metrics_inplace(self) -> None:
        df_reference = self.vector_reference.associations

        for idx, vector_noisy in enumerate(self.vectors_noisy):
            quality = {}
            df_noisy = vector_noisy.associations

            noisy_labels = vector_noisy.associations["label"].dropna().unique()
            for label_name in noisy_labels:
                quality[label_name] = {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                }

            for (record, label), df_reference_sub_record_label in df_reference.groupby(
                ["record", "label"]
            ):
                token_set_reference = util.get_token_range(
                    df_reference_sub_record_label
                )

                df_noisy_sub_record_label = df_noisy.loc[
                    (df_noisy["record"] == record) & (df_noisy["label"] == label)
                ].copy()

                if len(df_noisy_sub_record_label) > 0:
                    token_set_noisy = util.get_token_range(df_noisy_sub_record_label)

                    tps = len(token_set_reference.intersection(token_set_noisy))
                    fps = len(token_set_noisy.difference(token_set_reference))
                    fns = len(token_set_reference.difference(token_set_noisy))
                else:
                    tps = 0
                    fps = 0
                    fns = len(token_set_reference)

                quality[label]["true_positives"] += tps
                quality[label]["false_positives"] += fps
                quality[label]["false_negatives"] += fns
            self.vectors_noisy[idx].quality = quality.copy()

    def _set_quantity_metrics_inplace(self) -> None:
        df_noisy_vectors = common_util.get_all_noisy_vectors_df(self)
        df_noisy_vectors_flat = util.flatten_range_df(df_noisy_vectors)

        for source, df_noisy_vectors_flat_sub_source in df_noisy_vectors_flat.groupby(
            "source"
        ):
            quantity = {
                label: {
                    "record_coverage": len(
                        df_noisy_vectors_flat_sub_source.loc[
                            (df_noisy_vectors_flat_sub_source["label"] == label)
                        ].record.unique()
                    ),
                    "total_hits": df_noisy_vectors_flat_sub_source.loc[
                        (df_noisy_vectors_flat_sub_source["label"] == label)
                    ].beginner.sum(),
                    "source_overlaps": 0,
                    "source_conflicts": 0,
                }
                for label in df_noisy_vectors_flat_sub_source["label"].dropna().unique()
            }

            for (
                record,
                label,
            ), df_noisy_vectors_sub_record_label in df_noisy_vectors.loc[
                df_noisy_vectors["source"] == source
            ].groupby(
                ["record", "label"]
            ):
                df_noisy_vectors_without_source_sub_record = df_noisy_vectors.loc[
                    (df_noisy_vectors["source"] != source)
                    & (df_noisy_vectors["record"] == record)
                ]
                quantity = util.add_conflicts_and_overlaps(
                    quantity,
                    label,
                    df_noisy_vectors_sub_record_label,
                    df_noisy_vectors_without_source_sub_record,
                )

            for idx, vector in enumerate(self.vectors_noisy):
                if vector.identifier == source:
                    self.vectors_noisy[idx].quantity = quantity.copy()

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
                vector_stats["total_hits"] = quantity["total_hits"]
                vector_stats["source_conflicts"] = quantity["source_conflicts"]
                vector_stats["source_overlaps"] = quantity["source_overlaps"]
                statistics.append(vector_stats.copy())

        stats_df = pd.DataFrame(statistics)
        return stats_df

    def weakly_supervise(self) -> pd.Series:
        stats_df = self.quality_metrics()
        if len(stats_df) == 0:
            raise exceptions.MissingStatsException(
                "Empty statistics; can't compute weak supervision"
            )
        stats_lkp = stats_df.set_index(["identifier", "label_name"]).to_dict(
            orient="index"
        )  # pairwise [heuristic, label] lookup for precision

        enlm_df = pd.DataFrame(self.records, columns=["record"])
        enlm_df = enlm_df.set_index("record")

        for vector in self.vectors_noisy:
            vector_df = vector.associations
            vector_id = vector.identifier
            vector_df["prediction"] = vector_df.apply(
                lambda x: [
                    x["label"],
                    stats_lkp[(vector_id, x["label"])]["precision"] * (x["confidence"]),
                    set(list(range(x["chunk_idx_start"], x["chunk_idx_end"] + 1))),
                    x["chunk_idx_start"],
                ],
                axis=1,
            )
            vector_series = vector_df.set_index("record")[["prediction"]]
            enlm_df[vector.identifier] = np.empty((len(enlm_df), 0)).tolist()
            for idx, row in vector_series.iterrows():
                enlm_df[vector.identifier].loc[idx].append(row["prediction"])
        return enlm_df.apply(util._ensemble, axis=1)
