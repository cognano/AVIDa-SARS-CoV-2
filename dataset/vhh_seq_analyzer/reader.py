import polars as pl
from Bio import SeqIO


class FastaReader:
    """
    A class for reading FASTA files and converting them into a Polars DataFrame.

    Attributes
    ----------
    _files : list[str]
        A list of file paths to FASTA files to be read.
    """

    def __init__(self, files: list[str]):
        """
        Initializes the FastaReader with a list of FASTA file paths.

        Parameters
        ----------
        files : list[str]
            The paths to the FASTA files to be loaded.
        """
        self._files = files

    def _load(self, file: str) -> list[tuple[str, str, str]]:
        """
        Reads a FASTA file and extracts each record's ID, sequence, and description.

        Parameters
        ----------
        file : str
            Path to the FASTA file to be parsed.

        Returns
        -------
        list[tuple[str, str, str]]
            A list of tuples, each containing the ID, sequence, and description of a record.
        """
        return [(rec.id, str(rec.seq), rec.description) for rec in SeqIO.parse(file, "fasta")]

    def to_polars(self) -> pl.DataFrame:
        """
        Converts the specified FASTA files into a Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            A DataFrame with columns 'id', 'sequence', and 'description'.
        """
        # Generate a generator expression that aggregates all records from all files
        loaders = (record for file in self._files for record in self._load(file))

        # Temporarily enable Polars' string cache to optimize memory usage for string columns
        df = pl.DataFrame(
            loaders,
            schema={
                "id": pl.String,
                "sequence": pl.String,
                "description": pl.String,
            },
        )

        return df
