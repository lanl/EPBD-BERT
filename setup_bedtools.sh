#!/bin/bash

bedtools_dir=bedtools
mkdir -p $bedtools_dir
cd $bedtools_dir
wget https://github.com/arq5x/bedtools2/releases/download/v2.31.0/bedtools.static                   # we used version 2.31.0 (Aug 18 2023)
mv bedtools.static bedtools                                                                         # renaming
chmod a+x bedtools 

# bash setup_bedtools.sh