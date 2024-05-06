.. TODO
.. ========================================
.. * venv setup
.. * Installation (this repo)
.. * Preprocessed dataset setup
.. * Preprocess data from scratch
.. * Run (train+test) developed models
.. * Analysis

Welcome to EPBDxBERT's documentation!
======================================
This repository corresponds to the article titled as **Advancing Transcription Factor Binding Site Prediction Using DNA Breathing Dynamics and Sequence Transformers via Cross Attention**.


.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8222805.svg
   :target: https://www.biorxiv.org/content/10.1101/2024.01.16.575935v2


.. figure:: plots/EPBD_Arch.jpg
    :width: 50%
    :align: center
    
    Figure 1: Overview of the proposed **EPBDxBERT** framework


Resources
========================================
* `Paper <https://www.biorxiv.org/content/10.1101/2024.01.16.575935v2.abstract>`_
* `Code <https://github.com/ceodspspectrum/EPBD-BERT>`_
* `Documentation <https://github.io/ceodspspectrum/EPBD-BERT>`_
* `Analysis Notebooks <https://github.com/ceodspspectrum/EPBD-BERT/tree/main/analysis>`_



Installation
========================================
**Bedtools setup:** Follow `bedtools` installation guide from [here](<https://bedtools.readthedocs.io/en/latest/content/installation.html>). We also provide the installation script that downloads the pre-compiled binary of the software into the *bedtools* directory.

```bash
    bash setup_bedtools.sh    
    export PATH=$PATH:$(pwd)/bedtools
``` 
    
**virtual env:**
<!-- pyfastx, pandas, numpy -->
```bash
    conda activate .venv
```

Data preprocessing steps
========================================
    
```bash
    python data_preprocessing/0_download_data.py
    bash data_preprocessing/1_preprocess_narrowPeaks_and_humanGenome.sh
    sbatch data_preprocessing/2.1_compute_overlappings_job.sh
    bash data_preprocessing/3_postprocess.sh
    data_preprocessing/4_merge_peaks_with_same_labels.ipynb
    data_preprocessing/5.1_extract_bins_containingOtherThanACGT.ipynb
    bash data_preprocessing/5.2_compute_peaks_with_labels_clean.sh
    data_preprocessing/6.1_create_data_for_pydnaepbd.ipynb
    data_preprocessing/6.2_create_data_for_dnabert2.ipynb
    data_preprocessing/7_create_train_val_test_set.ipynb
    data_preprocessing/8_create_labels_dict.ipynb
```


