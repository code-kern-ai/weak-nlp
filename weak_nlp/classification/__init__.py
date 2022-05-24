import pandas as pd
import weak_nlp
from weak_nlp.classification import util

from weak_nlp.shared import common_util, exceptions


class ClassificationAssociation(weak_nlp.Association):
    pass


class CNLM(weak_nlp.NoisyLabelMatrix):
    def __init__(self, vectors: weak_nlp.SourceVector):
        super().__init__(vectors)

    def _set_quality_metrics_inplace(self) -> None:
        # There is one reference vector, which has been manually labeled (e.g. in the UI)
        # we compare that vector to all other N vectors (that is we make N comparisons).
        # This way, we can easily compute the quality of one noisy heuristic
        # We do so via joining the sets on which we have pairs, and compare the actual and noisy label
        reference_labels = list(
            self.vector_reference.associations["label"].dropna().unique()
        )
        for idx, vector_noisy in enumerate(self.vectors_noisy):
            if not vector_noisy.is_empty:
                quality = {}
                noisy_labels = list(
                    vector_noisy.associations["label"].dropna().unique()
                )
                for label_name in noisy_labels + reference_labels:
                    quality[label_name] = {
                        "true_positives": 0,
                        "false_positives": 0,
                    }

                df_inner_join = (
                    self.vector_reference.associations.set_index("record")
                    .join(
                        vector_noisy.associations.set_index("record"),
                        how="inner",
                        lsuffix="_reference",
                        rsuffix="_noisy",
                    )
                    .reset_index()
                )

                for label_name, df_grouped in df_inner_join.groupby("label_noisy"):
                    num_intersections = len(df_grouped)
                    true_positives = (
                        df_grouped["label_reference"] == df_grouped["label_noisy"]
                    ).sum()
                    false_positives = num_intersections - true_positives
                    quality[label_name] = {
                        "true_positives": true_positives,
                        "false_positives": false_positives,
                    }
                self.vectors_noisy[idx].quality = quality.copy()

    def _set_quantity_metrics_inplace(self) -> None:
        # We don't need the manually labeled reference vector for this; however,
        # we require all other heuristics of that task. We always look at one specific
        # heuristic (source), and compare all N-1 other heuristics against it
        df_noisy_vectors = common_util.get_all_noisy_vectors_df(self)
        for source, df_source in df_noisy_vectors.groupby("source"):
            df_without_source = df_noisy_vectors.loc[
                (df_noisy_vectors["source"] != source)
                & (df_noisy_vectors["record"].isin(df_source.record.unique()))
                # no need to load parts of other heuristics we don't care about for this heuristic
            ]
            quantity = {
                label: {
                    "record_coverage": len(df_source.loc[df_source["label"] == label]),
                    "source_overlaps": 0,
                    "source_conflicts": 0,
                }
                for label in df_source["label"].dropna().unique()
            }
            for record_series in df_without_source.groupby("record")["label"]:
                record_id, labels = record_series
                labels_unique = labels.unique()  # e.g. ["clickbait", "regular"]
                row_source = df_source.loc[df_source["record"] == record_id].iloc[0]
                label = row_source.label  # e.g. "clickbait"
                if label in labels_unique:
                    quantity[label]["source_overlaps"] += 1
                if len(labels_unique) > 1 or labels_unique[0] != label:
                    quantity[label]["source_conflicts"] += 1

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
                statistics.append(vector_stats.copy())

        stats_df = pd.DataFrame(statistics)
        if len(stats_df) > 0:
            stats_df["precision"] = stats_df.apply(common_util.calc_precision, axis=1)
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

    def weakly_supervise(self) -> pd.Series:
        stats_df = self.quality_metrics()
        if len(stats_df) == 0:
            raise exceptions.MissingStatsException(
                "Empty statistics; can't compute weak supervision"
            )
        stats_lkp = stats_df.set_index(["identifier", "label_name"]).to_dict(
            orient="index"
        )  # pairwise [heuristic, label] lookup for precision

        # We can collect *all* heuristic results for this noisy label matrix
        # and apply weight lookups for each prediction
        cnlm_df = pd.DataFrame(self.records, columns=["record"])
        cnlm_df = cnlm_df.set_index("record")
        for vector in self.vectors_noisy:
            vector_df = vector.associations
            if len(vector_df) > 0:
                vector_df["prediction"] = vector_df.apply(
                    lambda x: [
                        x["label"],
                        stats_lkp[(vector.identifier, x["label"])]["precision"]
                        * (x["confidence"]),
                    ],
                    axis=1,
                )
                vector_series = vector_df.set_index("record")["prediction"]
                cnlm_df[vector.identifier] = vector_series
        cnlm_df = cnlm_df.loc[~(cnlm_df.isnull()).all(axis=1)].fillna(
            "-"
        )  # hard to deal with np.nan

        return cnlm_df.apply(util._ensemble, axis=1)
