.. EPBD-BERT documentation master file, created by
   sphinx-quickstart on Mon Jan  8 21:08:38 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EPBD-BERT's documentation!
=====================================
This repository corresponds to the article titled **"Advancing Transcription Factor Binding Site Prediction Using DNA Breathing Dynamics and Sequence Transformers via Cross Attention"**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11130474.svg)](https://doi.org/10.5281/zenodo.11130474)
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

.. toctree::
   :maxdepth: 4

   modules/modules 
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
