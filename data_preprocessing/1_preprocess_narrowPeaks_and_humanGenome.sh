#!/bin/bash
export PATH=$PATH:$(pwd)/bedtools # bedtools software path

# inputs
inp_peak_files_dir=data/downloads/wgEncodeAwgTfbsUniform/ # wgEncodeAwgTfbsUniform_small or wgEncodeAwgTfbsUniform
inp_human_genome_file=data/downloads/hg19_latest/hg19.genome # this contains the chromosome sizes

# outputs
out_dir=data/processed/
out_peak_files_dir="$out_dir"wgEncodeAwgTfbsUniform_sorted/ # wgEncodeAwgTfbsUniform_sorted_small or wgEncodeAwgTfbsUniform_sorted
human_genome_filename="$(basename ${inp_human_genome_file})"
out_human_genome_file=$out_dir$human_genome_filename.windowed.sorted.gz

mkdir -p $out_peak_files_dir


#-------------------------Preprocessing narrowPeak files
echo "Sorting all the peak file in the '$inp_peak_files_dir'..."
if [ -z "$(ls -A $out_peak_files_dir)" ]; then
    for filepath in $inp_peak_files_dir*; do 
        if [[ $filepath == *.narrowPeak.gz ]]; then
            echo -e "\t$filepath";
            filename="$(basename ${filepath} .narrowPeak.gz)";
            # echo $filename;
            bedtools sort -i $filepath > $out_peak_files_dir$filename.narrowPeak.sorted
            gzip $out_peak_files_dir$filename.narrowPeak.sorted -f
        fi
    done
else
   echo -e "\t'$out_peak_files_dir'" contains data. Doing no preprocessing of narrowPeak files.
fi



#-------------------------Preprocessing human genome
window_size=200
step_size=$window_size

if [ -f $out_human_genome_file ]; then
    echo -e "\t'$out_human_genome_file' exists."
else 
    echo "Dividing human genome $window_size-bp bins with step-size $step_size..."  # window_size==step_size indicates no-overlapping
    bedtools makewindows -g $inp_human_genome_file -w $window_size -s $step_size > $out_dir$human_genome_filename.windowed #data/processed/hg19.genome.windowed
    bedtools sort -i $out_dir$human_genome_filename.windowed > $out_dir$human_genome_filename.windowed.sorted   # sorting 200-bp bins
    gzip $out_dir$human_genome_filename.windowed.sorted -f    # gzipping
    rm -rf $out_dir$human_genome_filename.windowed    # removing extra file
fi

#------------------------------------------------
# echo "Concatenating all the sorted peak file paths and names..."
# all_peakfile_paths=""
# filenames=""
# n_peakfiles_to_process=5
# i=1
# for filepath in $out_sorted_peak_files_dir*; do 
#     if [[ $filepath == *.narrowPeak.sorted.gz ]]; then
#         all_peakfile_paths+=" $filepath"
#         filenames+=" $(basename ${filepath} .narrowPeak.sorted.gz)"; # removing ext b/c files.txt has a column named "tableName" in this format.
        
#         # to use small number of peak files, ie 5
#         # echo $i

#         if [ $i == 5 ]; then
#             break
#         fi 
#         i=$(($i+1))
#     fi
# done
# echo -e "\t$all_peakfile_paths"
# echo -e "\t$filenames"


# echo "Intersecting the sorted-genome with the sorted-narrowPeak files..."
# bedtools intersect \
#     -a $windowed_sorted_human_genome_filepath \
#     -b $all_peakfile_paths \
#     -wo \
#     -f .5 \
#     -sorted \
#     -names $filenames > $out_final_filepath
# gzip $out_final_filepath -f