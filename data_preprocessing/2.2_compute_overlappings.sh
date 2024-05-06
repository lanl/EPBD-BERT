#!/usr/bin/sh

# run this file from data_preprocessing/2.1_job.sh
# to emulated a first single job, otherwise comment out
# SLURM_ARRAY_TASK_ID=0 
# echo $SLURM_ARRAY_TASK_ID

n_peak_files=690
n_jobs=115
n_peakfiles_to_process=$(( $n_peak_files / $n_jobs )) # by each independent node

start_idx=$(( $n_peakfiles_to_process*$SLURM_ARRAY_TASK_ID+1 ))
end_idx=$(( $n_peakfiles_to_process*($SLURM_ARRAY_TASK_ID+1) ))

echo $start_idx $end_idx

#-----------------------------------paths variables
export PATH=$PATH:$(pwd)/bedtools # bedtools software path

# path
data_root=data/processed/
sorted_peak_files_dir="$data_root"wgEncodeAwgTfbsUniform_sorted/ # wgEncodeAwgTfbsUniform_sorted_small or wgEncodeAwgTfbsUniform_sorted
windowed_sorted_human_genome_filepath="$data_root"hg19.genome.windowed.sorted.gz
out_peaks_dir="$data_root"peaks/

mkdir -p $out_peaks_dir

#-------------------------------------
function intersect(){
    all_peakfile_paths=$1
    filenames=$2
    
    echo "Intersecting the sorted-genome with the sorted-narrowPeak files ("$start_idx"-"$end_idx")..."
    # echo -e "\t$filenames"
    # echo -e "\t$all_peakfile_paths"

    bedtools intersect \
        -a $windowed_sorted_human_genome_filepath \
        -b $all_peakfile_paths \
        -wo \
        -f .5 \
        -sorted \
        -names $filenames > "$out_peaks_dir""$start_idx"_"$end_idx".sorted
    gzip "$out_peaks_dir""$start_idx"_"$end_idx".sorted -f
}

#--------------------------------
# this compiles a string containing n_peakfiles_to_process narrowPeak file paths to do the intersection with the human genome (windowed and sorted)
i=1
for filepath in $sorted_peak_files_dir*; do 
    if [[ $filepath == *.narrowPeak.sorted.gz ]]; then

        if [ $i -ge $start_idx ] && [ $i -le $end_idx ]; then
            # echo $i
            all_peakfile_paths+=" $filepath"
            filenames+=" $(basename ${filepath} .narrowPeak.sorted.gz)"; # removing ext b/c files.txt has a column named "tableName" in this format.
        fi

        i=$(( $i+1 ))
    fi
done
# echo $all_peakfile_paths
# echo $filenames

intersect "$all_peakfile_paths" "$filenames"