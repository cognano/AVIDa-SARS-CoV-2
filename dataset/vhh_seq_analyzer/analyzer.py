import sys

import numpy as np
import polars as pl
import scipy.stats as st

from .reader import FastaReader
from .types import DeconflictStrategy, Interation, TestPairs


class Analyzer:
    def __init__(self, fasta_files: list[str], min_occurrences: int = 2):
        """
        Initializes the Analyzer with fasta files.

        Parameters
        ----------
        fasta_files : list[str]
            List of paths to fasta files.
        min_occurrences : int, optional
            Minimum occurrences of a sequence.
            The default value of 2 means that singletons are removed.
        """
        self.min_occurrences = min_occurrences

        df = FastaReader(fasta_files).to_polars()
        self.df = df.select(
            pl.col("sequence"),
            self._extract_sample(df["description"]).alias("library"),
        )

    def _extract_sample(self, description: pl.Series) -> pl.Series:
        return description.str.extract(r"sample=([^;]+)").cast(pl.Categorical)

    def crosstab(self, filter_by_libraries: list[str] = []) -> pl.DataFrame:
        """
        Create a cross-tabulation of sequence occurrence for libraries.

        Parameters
        ----------
        filter_by_libraries : list[str], optional
            List of library names to filter the data.

        Returns
        -------
        pl.DataFrame
            A dataframe showing count of sequences in different libraries.
        """
        df = self.df

        # Filter by specified libraries
        if len(filter_by_libraries) > 0:
            df = df.filter(pl.col("library").is_in(filter_by_libraries))

        # Filter sequences based on minimum number of occurrences
        valid_sequences = (
            df["sequence"]
            .value_counts()
            .filter(pl.col("count") >= self.min_occurrences)
            .get_column("sequence")
        )
        df = df.filter(pl.col("sequence").is_in(valid_sequences))

        # Cross-tabulation
        df = df.pivot(
            values="sequence",
            index="sequence",
            columns="library",
            aggregate_function="len",
        ).fill_null(0)

        df = df.select(
            pl.col("sequence"), pl.col(sorted(df.select(pl.exclude("sequence")).columns))
        )

        return df

    def library_table(self, filter_by_libraries: list[str] = []) -> pl.DataFrame:
        """
        Create a table counting occurrences of each amino acid sequence across all libraries.

        Parameters
        ---
        filter_by_libraries
            List of library names to filter.

        Returns
        ---
        pl.DataFrame
            Table with sequences and their occurrences in each library, along with total occurrences.
        """
        df = self.crosstab(filter_by_libraries)

        # Add a total count column
        df = df.with_columns(df.select(pl.exclude("sequence")).sum_horizontal().alias("total"))

        # Add a sequential identifier (ONU) in descending order by total count
        df = df.sort(["total", "sequence"], descending=[True, False])
        df = df.with_columns(pl.Series("ONU", [f"ONU{str(i)}" for i in range(1, len(df) + 1)]))

        # Move the ONU, sequence, and total columns to the front
        df = df.select(pl.col("ONU", "sequence", "total"), pl.exclude("ONU", "sequence", "total"))

        return df

    def signed_p_value_table(self, test_pairs: TestPairs) -> pl.DataFrame:
        """
        Generates a table with signed p-values for each pair of 'mother' and 'target'
        groups provided in TestPairs. A signed p-value is used to represent the
        directionality of differences (increase or decrease), along with the
        statistically significant differences between each pair of groups.

        Parameters
        ----------
        test_pairs : TestPairs
            A collection of TestPair objects, each representing a pair of 'mother' and
            'target' groups to be compared.

        Returns
        -------
        pl.DataFrame
            A DataFrame where each column corresponds to a 'target' group from the
            provided test pairs. The column values are signed p-values indicating the
            statistical significance and direction of difference between the 'mother'
            and 'target' groups. Positive values indicate an increase in the 'target'
            group relative to 'mother', while negative values indicate a decrease.
        """
        all_libraries = test_pairs.flatten()
        df_libtable = self.library_table(all_libraries)

        return self._signed_p_value_table(df_libtable, test_pairs)

    def _signed_p_value_table(
        self, library_table: pl.DataFrame, test_pairs: TestPairs
    ) -> pl.DataFrame:
        signed_p_cols = []

        for pair in test_pairs:
            column = pl.concat(
                [
                    self._calc_p_value(library_table, pair.mother, pair.target),
                    self._identify_increase(library_table, pair.mother, pair.target),
                ],
                how="horizontal",
            ).select(
                pl.when(pl.col("increase"))
                .then(pl.col("revised_p_value"))
                .otherwise(-pl.col("revised_p_value"))
                .alias(pair.target)
            )

            signed_p_cols.append(column)

        df_signed_p = pl.concat(signed_p_cols, how="horizontal")
        df_sorted_signed_p = df_signed_p.select(pl.col(sorted(df_signed_p.columns)))
        return df_sorted_signed_p

    def annotate(
        self,
        control_pairs: TestPairs,
        treatment_pairs: TestPairs,
        readonly_pairs: TestPairs = [],
        significance_level: float = 0.05,
    ) -> pl.DataFrame:
        """
        Annotates sequences with labels based on statistical comparisons between control,
        treatment, and optionally, readonly pairs. It generates a DataFrame that includes
        signed p-values for comparisons, selects samples based on these values, and
        finally labels the sequences based on predefined criteria.

        Parameters
        ----------
        control_pairs : TestPairs
            Test pairs representing control groups.

        treatment_pairs : TestPairs
            Test pairs representing treatment groups.

        readonly_pairs : TestPairs, optional
            Test pairs representing additional groups not directly involved in the
            statistical comparisons but included for context based on the wet experiment.

        significance_level : float, optional
            The significance level used for determining statistical significance.

        Returns
        -------
        pl.DataFrame
            A DataFrame containing the original sequence data and labels indicating the type
            of interaction identified based on the predefined criteria.
        """
        all_pairs = control_pairs + treatment_pairs + readonly_pairs
        df_libtable = self.library_table(all_pairs.flatten())

        # Calculate signed p-values
        df_signed_p = self._signed_p_value_table(df_libtable, all_pairs)

        # Select samples based on signed p-values
        controls = control_pairs.flatten("target")
        treatments = treatment_pairs.flatten("target")
        sampled_control = df_signed_p.select(controls).map_rows(self._select_sample)
        sampled_treatment = df_signed_p.select(treatments).map_rows(self._select_sample)

        # Combine control and treatment samples
        df_sample = pl.concat(
            [
                sampled_control.rename({"map": "control"}),
                sampled_treatment.rename({"map": "treatment"}),
            ],
            how="horizontal",
        ).fill_null(-1)

        # Apply predefined criteria
        labels = np.vectorize(self.identify_interactions)(
            df_sample["control"], df_sample["treatment"], significance_level
        )

        # Create labeled DataFrame
        df_labeled = pl.DataFrame(
            [
                df_libtable["ONU"],
                df_libtable["sequence"],
                df_sample["control"],
                df_sample["treatment"],
                pl.Series("label", labels),
            ],
            schema={
                "ONU": pl.Utf8,
                "sequence": pl.Utf8,
                "control": pl.Float64,
                "treatment": pl.Float64,
                "label": pl.Utf8,
            },
        )

        return df_labeled

    def deconflict(
        self, dataframes: dict[str, pl.DataFrame], strategy: DeconflictStrategy
    ) -> dict[str, pl.DataFrame]:
        """
        Resolves label conflicts between multiple DataFrames based on a specified strategy.
        This method identifies sequences with conflicting labels (e.g., binder vs. non-binder),
        and applies the resolution strategy (e.g., relabeling binder to non-binder) to ensure
        consistency across the dataset.

        Parameters
        ----------
        dataframes : dict[str, pl.DataFrame]
            A dictionary where each key-value pair consists of a string key representing the
            name of the DataFrame and the DataFrame itself. These DataFrames should have
            'sequence' and 'label' columns for the comparison.

        strategy : DeconflictStrategy
            The strategy to be used for resolving label conflicts.

        Returns
        -------
        dict[str, pl.DataFrame]
            A dictionary of DataFrames, identical in structure to the input but with label
            conflicts resolved according to the provided strategy.
        """

        # Extract the specific labels to be changed from the strategy
        from_label = strategy.relabel_from
        to_label = strategy.relabel_to

        # Dictionary to hold the DataFrames after conflict resolution
        resolved_dataframes: dict[str, pl.DataFrame] = {}

        # Define a filter for relevant labels (e.g., 'binder', 'non-binder')
        relevant_labels_filter = pl.col("label").is_in(["binder", "non-binder"])

        # Iterate through all pairs of DataFrames to find and resolve conflicts
        for source_name, source_df in dataframes.items():
            for target_name, target_df in dataframes.items():
                # Skip comparing the DataFrame with itself
                if source_name == target_name:
                    continue

                # Identify sequences with conflicting labels between the current pair of DataFrames
                conflicts = (
                    source_df.filter(relevant_labels_filter)
                    .join(
                        target_df.filter(relevant_labels_filter),
                        on="sequence",
                        how="inner",
                        suffix="_r",
                    )
                    .filter(pl.col("label") != pl.col("label_r"))
                )

                # Select sequences to be relabeled according to the strategy
                conflict_sequences = conflicts.filter(pl.col("label") == from_label)["sequence"]

                # Update labels for sequences identified to have conflicts
                resolved_df = source_df.with_columns(
                    new_label=pl.when(pl.col("sequence").is_in(conflict_sequences))
                    .then(pl.lit(to_label))
                    .otherwise(pl.col("label"))
                )

                # Apply the updated labels, replacing the old 'label' column
                source_df = resolved_df.select(
                    pl.exclude("label", "new_label"), pl.col("new_label").alias("label")
                )

            # Store the deconflicted DataFrame
            resolved_dataframes[source_name] = source_df

        return resolved_dataframes

    def _calc_p_value(
        self, library_table: pl.DataFrame, mother: str | list[str], target: str | list[str]
    ) -> pl.DataFrame:
        """
        Calculates p-values for comparison between 'mother' and 'target' libraries.

        Parameters
        ----------
        library_table : pl.DataFrame
            The DataFrame containing the data to be analyzed. Expected to have columns
            that match the names provided in 'mother' and 'target'.

        mother : str | list[str]
            The names of the columns representing the reference groups.

        target : str | list[str]
            The names of the columns representing the experimental groups.

        Returns
        -------
        pl.DataFrame
            A DataFrame containing the calculated p-values and related statistical metrics
            for each sequence's comparison between 'mother' and 'target' libraries.
        """

        # Aggregate data for 'mother' and 'target' groups by summing up across the horizontal axis
        mother_series = library_table.select(pl.col(mother)).sum_horizontal().alias("mother")
        target_series = library_table.select(pl.col(target)).sum_horizontal().alias("target")
        compare_table = pl.DataFrame([mother_series, target_series])

        # Calculate pooled proportions, standard errors, and z-scores for comparison
        p_val_df = (
            compare_table.with_columns(
                pooled_p=(
                    (pl.col("mother") + pl.col("target")) / (pl.sum("mother") + pl.sum("target"))
                ),
            )
            .with_columns(
                p1=pl.col("mother") / pl.sum("mother"),
                p2=pl.col("target") / pl.sum("target"),
                pooled_se=(
                    pl.col("pooled_p")
                    * (1 - pl.col("pooled_p"))
                    * (1 / pl.sum("mother") + 1 / pl.sum("target"))
                ).sqrt(),
            )
            .with_columns(z_score=((pl.col("p1") - pl.col("p2")) / pl.col("pooled_se")))
            .with_columns(
                p_value=pl.col("z_score")
                .abs()
                .map_batches(lambda abs_z: st.norm.cdf(-abs_z) + (1 - st.norm.cdf(abs_z)))
            )
            .with_columns(
                # Adjust p-values rounded to 0.0 to the smallest float
                revised_p_value=pl.when(pl.col("p_value") == 0.0)
                .then(sys.float_info.min)
                .otherwise(pl.col("p_value"))
            )
        )

        return p_val_df

    def _identify_increase(
        self, library_table: pl.DataFrame, mother: str | list[str], target: str | list[str]
    ) -> pl.DataFrame:
        """
        Identifies whether there is an increase in proportion for the 'target' group
        compared to the 'mother' group.

        Parameters
        ----------
        library_table : pl.DataFrame
            The DataFrame containing the data to be analyzed. Expected to have columns
            that match the names provided in 'mother' and 'target'.

        mother : str | list[str]
            The names of the columns representing the reference groups.

        target : str | list[str]
            The names of the columns representing the experimental groups.

        Returns
        -------
        pl.DataFrame
            A DataFrame with a single column named 'increase', consisting of boolean values
            indicating whether there is an increase in the proportion of the 'target'
            group compared to the 'mother' group.
        """
        mother_series = library_table.select(pl.col(mother)).sum_horizontal().alias("mother")
        target_series = library_table.select(pl.col(target)).sum_horizontal().alias("target")
        compare_table = pl.DataFrame([mother_series, target_series])

        return compare_table.select(
            increase=(pl.col("mother") / pl.sum("mother") < pl.col("target") / pl.sum("target"))
        )

    def identify_interactions(self, control, treatment, threshold) -> Interation:
        """
        Identifies the type of interaction based on the control, treatment, and threshold values.
        """
        if abs(treatment) > threshold:
            return "non-significant"

        if treatment <= 0:
            return "non-binder"

        if abs(control) <= threshold and control > 0:
            return "noise"

        if control > 0 and control / treatment < 10**2.5:
            return "non-significant"

        return "binder"

    def _select_sample(self, t: tuple):
        s = pl.Series(t)
        if s.max() > 0:
            return s.filter(s > 0).min()
        return s.filter(s <= 0).max()
