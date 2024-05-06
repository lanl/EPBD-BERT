export PATH=$PATH:$(pwd)/bedtools # bedtools software path

data_root=data/processed/

echo Removing peaks that contains other than ACGT...
# 13,277,862 - 1,003,451 = 12,274,411
bedtools subtract \
    -a "$data_root"generic_negative_regions.sorted.tsv.gz \
    -b "$data_root"generic_negative_regions_issues.sorted.tsv.gz > "$data_root"generic_negative_regions_clean.txt


echo Sorting...
bedtools sort -i "$data_root"generic_negative_regions_clean.txt > "$data_root"generic_negative_regions_clean.tsv

echo Gzipping...
gzip "$data_root"generic_negative_regions_clean.tsv -f

rm -rf "$data_root"generic_negative_regions_clean.txt

# to run this: sh data_preprocessing/9.3_clean_generic_regions.sh