# Welcome to EPBDxBERT's Documentation!

This repository corresponds to the article titled **"Advancing Transcription Factor Binding Site Prediction Using DNA Breathing Dynamics and Sequence Transformers via Cross Attention"**.

![DOI badge](https://zenodo.org/badge/DOI/10.5281/zenodo.8222805.svg)
(https://www.biorxiv.org/content/10.1101/2024.01.16.575935v2)

![EPBDxBERT Framework](plots/EPBD_Arch.jpg)
*Figure 1: Overview of the proposed EPBDxBERT framework*

Understanding the impact of genomic variants on transcription factor binding and gene regulation remains a key area of
research, with implications for unraveling the complex mechanisms underlying various functional effects. This software
framework delves into the role of DNA's biophysical properties, including thermodynamic stability, shape, and flexibility in
transcription factor (TF) binding. In this library, we have developed a multi-modal deep learning model integrating these
properties with DNA sequence data. Trained on ChIP-Seq (chromatin immunoprecipitation sequencing) data in-vivo
involving 690 TF-DNA binding events in human genome, our model significantly improves prediction performance in over
660 binding events, with up to 9.6% increase in AUROC metric compared to the baseline model when using no DNA
biophysical properties explicitly. Further, we expanded our analysis to in-vitro high-throughput Systematic Evolution
of Ligands by Exponential enrichment (SELEX) dataset, comparing our model with
established frameworks. The inclusion of EPBD features consistently improved TF binding predictions across different cell
lines in these datasets. Notably, for complex ChIP-Seq datasets, integrating DNABERT2 with a cross-attention mechanism
provided greater predictive capabilities and insights into the mechanisms of disease-related non-coding variants found in
genome-wide association studies. This work highlights the importance of DNA biophysical characteristics in TF binding
and the effectiveness of multi-modal deep learning models in gene regulation studies

## Resources

- [Paper](https://www.biorxiv.org/content/10.1101/2024.01.16.575935v2.abstract)
- [Code](https://github.com/ceodspspectrum/EPBD-BERT)
- [Documentation](https://github.io/ceodspspectrum/EPBD-BERT)
- [Analysis Notebooks](https://github.com/ceodspspectrum/EPBD-BERT/tree/main/analysis)

## Installation

### Bedtools Setup

Follow the [bedtools installation guide](https://bedtools.readthedocs.io/en/latest/content/installation.html). We also provide a script that downloads the pre-compiled binary of the software into the *bedtools* directory:

```bash
bash setup_bedtools.sh
export PATH=$PATH:$(pwd)/bedtools
```

## Data Preprocessing Steps

```bash
python data_preprocessing/0_download_data.py
bash data_preprocessing/1_preprocess_narrowPeaks_and_humanGenome.sh
sbatch data_preprocessing/2.1_compute_overlappings_job.sh
bash data_preprocessing/3_postprocess.sh
jupyter notebook data_preprocessing/4_merge_peaks_with_same_labels.ipynb
jupyter notebook data_preprocessing/5.1_extract_bins_containingOtherThanACGT.ipynb
bash data_preprocessing/5.2_compute_peaks_with_labels_clean.sh
jupyter notebook data_preprocessing/6.1_create_data_for_pydnaepbd.ipynb
jupyter notebook data_preprocessing/6.2_create_data_for_dnabert2.ipynb
jupyter notebook data_preprocessing/7_create_train_val_test_set.ipynb
jupyter notebook data_preprocessing/8_create_labels_dict.ipynb
```


## Authors

* [Anowarul Kabir] (mailto:akabir4@gmu.edu)- Computer Sciece, George Mason University
* [Manish Bhattarai] (mailto:ceodspspectrum@lanl.gov)- Theoretical Division, Los Alamos National Laboratory
* [Kim Rasmussen] (mailto:kor@lanl.gov)- Theoretical Division, Los Alamos National Laboratory
* [Amarda Shehu] (mailto:ashehu@gmu.edu)- Computer Sciece, George Mason University
* [Anny Usheva] (mailto:Anny Usheva@brown.edu>)-Surgery, Rhode Island Hospital and Brown University
* [Alan Bishop] (mailto:arb@lanl.gov)- Theoretical Division, Los Alamos National Laboratory
* [Boian S. Alexandrov] (mailto:boian@lanl.gov)- Theoretical Division, Los Alamos National Laboratory

## How to cite EPBD-BERT?
```latex
@article{kabir2024advancing,
  title={Advancing Transcription Factor Binding Site Prediction Using DNA Breathing Dynamics and Sequence Transformers via Cross Attention},
  author={Kabir, Anowarul and Bhattarai, Manish and Rasmussen, Kim {\O} and Shehu, Amarda and Bishop, Alan R and Alexandrov, Boian and Usheva, Anny},
  journal={bioRxiv},
  pages={2024--01},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Acknowledgements

Los Alamos National Lab (LANL), T-1

## Copyright notice

© 2024. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

LANL O#4717

## License

This program is Open-Source under the BSD-3 License.
 
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

