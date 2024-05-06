export PATH=$PATH:$(pwd)/bedtools # bedtools software path

data_root=data/processed/

# regions that do not overlapp with any peaks from 690 tf-dna binding peaks are considered as generic negative regions.
# (200 bp windowed human genome)15,181,531 - (pos 200bp regions with 50% overlapping regions)1,903,669 = 13,277,862
# this takes ~7 seconds
bedtools subtract \
    -a "$data_root"hg19.genome.windowed.sorted.gz \
    -b "$data_root"peaks_with_labels_clean.tsv.gz \
    -f 0.5 \
    -sorted > "$data_root"generic_negative_regions.sorted.tsv

echo Gzipping...
gzip "$data_root"generic_negative_regions.sorted.tsv -f