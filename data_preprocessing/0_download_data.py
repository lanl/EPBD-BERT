import os
import subprocess
import pandas as pd

hg19_download_dir = "data/downloads/hg19_latest/"
peaks_download_dir = "data/downloads/wgEncodeAwgTfbsUniform/"
os.makedirs(hg19_download_dir, exist_ok=True)
os.makedirs(peaks_download_dir, exist_ok=True)

# download uniform peak files from http://hgdownload.soe.ucsc.edu/goldenpath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/
subprocess.run(
    f"rsync -a -P rsync://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/ {peaks_download_dir}",
    shell=True,
)

# download human genome assembly GRCh37/hg19
subprocess.run(
    f"rsync -a -P rsync://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.chrom.sizes {hg19_download_dir}",
    shell=True,
)
subprocess.run(
    f"rsync -a -P rsync://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz {hg19_download_dir}",
    shell=True,
)

# considering chromosomes 1-22 and X.
chr_size_df = pd.read_csv(
    f"{hg19_download_dir}hg19.chrom.sizes", sep="\t", header=None, names=["chr", "size"]
)
chrs = ["chr" + str(i) for i in range(1, 23)] + ["chrX"]
chr_size_df = chr_size_df[chr_size_df["chr"].isin(chrs)]
chr_size_df.to_csv(
    f"{hg19_download_dir}hg19.genome", sep="\t", header=False, index=False
)
