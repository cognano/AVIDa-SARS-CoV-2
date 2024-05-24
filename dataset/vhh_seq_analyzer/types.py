import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, RootModel

Pair = Literal["mother", "target"]

Interation = Literal[
    "binder",
    "non-binder",
    "non-significant",
    "noise",
]


class TestPair(BaseModel):
    """
    Represents a pair of libraries (mother and target) used in statistical test,
    such as calculating Z-scores or p-values.

    Attributes
    ----------
    mother : str | list[str]
        The name(s) of the reference library used in the statistical comparison.
        This could represent a baseline data. When multiple names are provided,
        their data is combined before analysis.
    target : str
        The name of the test group being compared against the mother. This is
        typically the group subjected to the experimental condition.

    """
    mother: str | list[str]
    target: str


class TestPairs(RootModel[list[TestPair]]):
    """
    A collection of TestPair objects.
    """
    root: list[TestPair]

    def __iter__(self):
        """
        Allows for iteration over the collection of TestPair objects.
        """
        return iter(self.root)

    def __getitem__(self, index):
        """
        Allows for index-based access to the collection of TestPair objects.

        Parameters
        ----------
        index : int
            The index of the element to retrieve.
        """
        return self.root[index]

    def __add__(self, other: "TestPairs") -> "TestPairs":
        """
        Combines two TestPairs collections into a single one, enabling the
        accumulation of test pairs.

        Parameters
        ----------
        other : TestPairs
            TestPairs to combine.
        """
        return TestPairs(self.root + other.root)

    def flatten(self, filter_key: Pair = "") -> list[str]:
        """
        Creates a flat list of all libraries (either 'mother' or 'target',
        or both) from the TestPair objects in the collection.

        Parameters
        ----------
        filter_key : {"mother", "target"}, optional
            filter by the key.

        Returns
        -------
        list[str]
            flattened list
        """
        values = set()

        for pair in self.root:
            for key, value in pair.model_dump().items():
                if filter_key == "" or filter_key == key:
                    if isinstance(value, list):
                        values.update(value)
                    else:
                        values.add(value)
        return list(values)


class DeconflictStrategy(BaseModel):
    """
    Defines a strategy for resolving label conflicts for the same amino acid
    sequences across different analysis units.

    Attributes
    ----------
    strategy_type : Literal["relabel"]
        The type of deconflict strategy. Currently, 'relabel' is the only
        supported strategy, which involves changing the label of the amino acid
        sequence to a specified new label.
    relabel_from : str
        The original label that needs to be changed to avoid conflict.
    relabel_to : str
        The new label to replace the original.
    """

    strategy_type: Literal["relabel"] = Field(..., alias="type")
    relabel_from: str = Field(..., alias="from")
    relabel_to: str = Field(..., alias="to")


class AnalysisUnit(BaseModel):
    """
    Represents an independent analysis task designed to process a subset of data and
    produce statistical analysis results. The analysis is focused on comparing
    control and treatment test pairs, with additional context provided by readonly test pairs.
    Results are output to a CSV file using the specified unit name.

    Attributes
    ----------
    name : str
        The name used for the CSV file containing the analysis results.
    fixedColumns : dict
        Specifies columns with constant values to be included in the CSV output.
        Each key-value pair defines a column name and its fixed value, providing
        essential context or metadata for the analysis.
    control : TestPairs
        The control group test pairs, serving as the baseline for statistical
        comparisons to identify significant differences.
    treatment : TestPairs
        The treatment group test pairs, which are compared against the control
        to assess the effect of the experimental condition.
    readonly : TestPairs
        Test pairs that are not directly analyzed but are used to identify and
        exclude singleton amino acid sequences before statistical testing. This
        process can influence the presence ratios of amino acid sequences and
        potentially affect p-values.
    """
    name: str
    fixedColumns: dict
    control: TestPairs
    treatment: TestPairs
    readonly: TestPairs

    def list_pairs(self):
        return self.control + self.treatment + self.readonly


class AnnotateConfig(BaseModel):
    """
    Configuration for annotation processes.

    Attributes
    ----------
    name : str
        The name of the annotation configuration.
    dataDirs : list[str]
        The directories containing the data relevant to the annotation tasks.
    deconflictStrategy : DeconflictStrategy
        The strategy to be used for resolving label conflicts across the analysis units.
    analysisUnits : list[AnalysisUnit]
        A list of independent analysis units to perform separate statistical analyses.
    """
    config_path: Optional[str] = None
    config_dir: Optional[str] = None
    name: str
    dataDirs: list[str]
    outDir: str = Field(default="./out")
    deconflictStrategy: DeconflictStrategy
    analysisUnits: list[AnalysisUnit] = Field(min_length=1)

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "AnnotateConfig":
        """
        Load an instance of AnnotateConfig from a YAML file.
        """
        abs_path = os.path.abspath(yaml_path)
        with open(abs_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        instance = cls.model_validate(yaml_data)
        instance.config_path = abs_path
        instance.config_dir = os.path.dirname(abs_path)
        return instance

    def list_libraries(self) -> list[str]:
        """
        Generates a list of all libraries across all analysis units.
        """
        libraries = set()
        for test in self.analysisUnits:
            libraries.update(test.list_pairs().flatten())

        return list(libraries)

    def validate_data_dirs(self, src_path: Optional[str] = None) -> list[Path]:
        """
        Validates and returns the absolute paths to the data directories, raising
        an error if any of them do not exist.
        """
        if src_path is None:
            src_path = self.config_dir

        if src_path is None:
            raise ValueError("Source path is not specified.")

        data_dirs = []
        for data_dir in self.dataDirs:
            abs_path = os.path.abspath(os.path.join(src_path, data_dir))
            data_dir_path = Path(abs_path)

            if not data_dir_path.exists():
                raise FileNotFoundError(f"The specified data directory does not exist: {data_dir_path}")

            data_dirs.append(data_dir_path)

        return data_dirs

    def validate_out_dir(self, src_path: Optional[str] = None) -> Path:
        """
        Validates and returns the absolute path to the output directory, creating
        it if it does not exist.
        """
        if src_path is None:
            src_path = self.config_dir

        if src_path is None:
            raise ValueError("Source path is not specified.")

        out_dir = Path(os.path.abspath(os.path.join(src_path, self.outDir)))

        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        return out_dir

    def find_files_by_extension(self, extension: str) -> list[str]:
        """
        Recursively searches for files with the specified extension in the data directories
        and returns a list of their absolute paths.
        """
        files = []
        data_dirs = self.validate_data_dirs(self.config_dir)

        for data_dir in data_dirs:
            for root, _, filenames in os.walk(data_dir):
                for filename in filenames:
                    if filename.endswith(extension):
                        abs_path = os.path.join(root, filename)
                        files.append(abs_path)


        return files
