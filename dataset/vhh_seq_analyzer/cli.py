import os

import polars as pl

from .analyzer import Analyzer
from .types import AnnotateConfig


class CLI(object):
    def annotate(self, config_file: str):
        # Load the configuration from the YAML file
        config = AnnotateConfig.load_from_yaml(config_file)
        out_dir = config.validate_out_dir()

        # Find all files with the "fasta" extension in the data directories
        files = config.find_files_by_extension("fasta")
        analyzer = Analyzer(files)

        # Initialize dictionaries to store dataframes and fixed columns
        dfs: dict[str, pl.DataFrame] = {}
        fixed_columns: dict[str, dict[str, str]] = {}

        # Iterate over each analysis unit in the configuration
        for unit in config.analysisUnits:
            df_labeled = analyzer.annotate(unit.control, unit.treatment, unit.readonly)
            dfs[unit.name] = df_labeled
            fixed_columns[unit.name] = unit.fixedColumns

        # Deconflict the dataframes based on the specified strategy
        deconflicted_dfs = analyzer.deconflict(dfs, config.deconflictStrategy)

        # Write the deconflicted dataframes to CSV files
        for k, df in deconflicted_dfs.items():
            out_file = out_dir.joinpath(f"{k}.csv")
            fixed_columns_expr = [
                pl.lit(v).alias(k) for k, v in fixed_columns[k].items()
            ]

            df.filter(pl.col("label").is_in(["binder", "non-binder"])).with_columns(
                fixed_columns_expr
            ).select(
                pl.col("sequence").alias("VHH_sequence"),
                pl.col("Ag_label"),
                pl.col("label").replace({"binder": 1, "non-binder": 0}).alias("label"),
                pl.col("subject_species", "subject_name", "subject_sex"),
            ).write_csv(out_file)

            print(f"Saved file to {out_file}")
