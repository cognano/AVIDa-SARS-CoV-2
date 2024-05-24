#!/bin/bash

set -eu

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
RAW_DIR="$REPO_ROOT/dataset/raw"
mkdir -p $RAW_DIR/fastq $RAW_DIR/fasta
OUT_DIR="$REPO_ROOT/dataset/out"
mkdir -p $OUT_DIR

IMAGE="vhh_constructor:latest"
VHH_CONSTRUCTOR="docker run --rm -v $RAW_DIR:/work/raw -v $PWD/out:/work/out $IMAGE construct_VHHs.sh"

# Construct VHH sequences
for file in $(ls $RAW_DIR/fastq); do
	if [[ $file == *R1* ]]; then
		R1File="/work/raw/fastq/$file"
		R2File="/work/raw/fastq/${file/R1/R2}"
		LibraryName=$(echo $file | sed -E 's/(.*)_S[0-9]+_L001_R1_001.fastq.gz/\1/')

		if [ -f $RAW_DIR/fasta/$LibraryName.fasta ]; then
			continue
		fi

		echo "Constructing VHHs for $LibraryName"
		$VHH_CONSTRUCTOR $R1File $R2File $LibraryName > $RAW_DIR/fasta/$LibraryName.fasta
	fi
done

# Create labeled data
echo "Creating labeled data"
python $REPO_ROOT/dataset/analyzer.py annotate $REPO_ROOT/dataset/config/OC43_bead.yaml
