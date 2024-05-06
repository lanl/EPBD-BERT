#!/usr/bin/sh

export PATH=$PATH:$(pwd)/bedtools # bedtools software path

data_root=data/processed/

echo Removing peaks that contains other than ACGT...
bedtools subtract -a "$data_root"peaks_with_labels.tsv.gz -b "$data_root"peaks_with_labels_issues.tsv > "$data_root"peaks_with_labels_clean.txt

echo Sorting...
bedtools sort -i "$data_root"peaks_with_labels_clean.txt > "$data_root"peaks_with_labels_clean.tsv

echo Gzipping...
gzip "$data_root"peaks_with_labels_clean.tsv -f

rm -rf "$data_root"peaks_with_labels_clean.txt