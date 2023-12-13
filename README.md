
# <div align="center">SPOT-1D-LM-Extended</div>
### <div align="center">Bolutito Babatunde and Madeline Davis</div>
#### <div align="center">Carnegie Mellon University</div>

Just as human language is composed of words, protein sequences consist of amino acids, forming the basis of biological processes. Understanding the relationship between a protein's structure and function is pivotal for advancements in healthcare and biotechnology. However, the complexity of proteins, influenced by factors such as folding patterns and evolutionary variations, poses a significant challenge. Recent developments, especially in deep learning like AlphaFold, have revolutionized protein structure prediction, offering more accuracy even for proteins without known homologs. This work focuses on single-sequence-based prediction methods, leveraging SPOT-1D-LM for predicting protein structural properties. By integrating embeddings from pre-trained models like ESM-1b and ProteinBERT, we will enhance our understanding of proteins' structural and functional motifs, crucial for addressing the protein-function prediction problem.

*Checkpoint given upon request*

System Requirements
----
Requirements taken from [SPOT-1D-LM Repo](https://github.com/jas-preet/SPOT-1D-LM)

**Hardware Requirements:**
SPOT-1D-LM predictor has been tested on standard ubuntu 18 computer with approximately 32 GB RAM to support the in-memory operations.

* [Python3.7](https://docs.python-guide.org/starting/install3/linux/)
* [Anaconda](https://anaconda.org/anaconda/virtualenv)
* [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive) (Optional if using GPU)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn) (Optional if using GPU)

Installation
----
Installation taken from [SPOT-1D-LM Repo](https://github.com/jas-preet/SPOT-1D-LM)

To install SPOT-1D-LM and it's dependencies following commands can be used in terminal:

1. `git clone https://github.com/jas-preet/SPOT-1D-LM.git`
2. `cd SPOT-1D-LM`

To download the model check pointsfrom the dropbox use the following commands in the terminal:

3. `wget https://servers.sparks-lab.org/downloads/SPOT-LM-checkpoints.xz`
4. `tar -xvf SPOT-LM-checkpoints.xz`

To download model data directly from website, click [here](http://zhouyq-lab.szbl.ac.cn/download/)

To install the dependencies and create a conda environment use the following commands

5. `conda create -n spot_1d_lm python=3.7`
6. `conda activate spot_1d_lm`

if GPU computer:
7. `conda install pytorch==1.7.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch`

for CPU only 
7. `conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch`

8. `pip install fair-esm`

9. `conda install pandas=1.1.1`

10. `conda install tqdm`

11. `pip install bio-embeddings[all]
`

Directory Detail
----

1. Once data is downloaded it should be in `spot_1d_lm` folder
2. Run `convert_labels_to_npy.ipynb` to download labels, i.e. `convert_data_to_numpy("casp12.txt")`
3. `dataset` folder contains data loader files for baseline and extension
4. `models_train` folder contains extended models
5. `models` folder contains baseline models
6. `inputs` folder contains example of `.fasta`, esm embeddings, and prottrans embeddings
7. `spot_1d_lm` folder should include `labels` folder you create for numpy labels
8. `results` folder should contain results from test sets after you run `run_SPOT-1D-LM.sh`


Execute Extension
----
To run Extension refer to the following file

`run_EXTENSION.sh` this contains instructions for generating embeddings and run training using best model parameters

`python generate_esm_extension.py --file_list $1 --device $2` can be run once SPOT-1D-LM dependencies are downloaded

`python generate_proteinbert_extension.py --file_list $1 --device $3` is trickier. Follow setup instructions from [ProteinBERT Repo](https://github.com/nadavbra/protein_bert), copy file into repo and run command on terminal

`python run_inference_extension.py` to run Extension model training, assuming correct directory instructions.


Execute Baseline
----
To run SPOT-1D-LM use the following command

`bash run_SPOT-1D-LM.sh file_lists/test_file_list.txt cpu cpu cpu` to run model, ESM-1b and ProtTrans on cpu

or 

`bash run_SPOT-1D-LM.sh file_lists/test_file_list.txt cpu cpu cuda:0` to run model on gpu and, ESM-1b and ProtTrans on cpu

or

`bash run_SPOT-1D-LM.sh file_lists/test_file_list.txt cuda:0 cuda:1 cuda:2` to run model, ESM-1b and ProtTrans on gpu


Citation Guide
----
for more details on this work refer the manuscript

Please also cite and refer to SPOT-1D-LM, ProteinBERT, ESM-1b, and ProtTrans as the input used in this work is from these works. 

```bibtex
@article{singh_SPOT_1D_LM_2022,
	title = {Reaching alignment-profile-based accuracy in predicting protein secondary and tertiary structural properties without alignment},
	volume = {12},
	copyright = {2022 The Author(s)},
	issn = {2045-2322},
	url = {https://www.nature.com/articles/s41598-022-11684-w},
	doi = {10.1038/s41598-022-11684-w},
	abstract = {Protein language models have emerged as an alternative to multiple sequence alignment for enriching sequence information and improving downstream prediction tasks such as biophysical, structural, and functional properties. Here we show that a method called SPOT-1D-LM combines traditional one-hot encoding with the embeddings from two different language models (ProtTrans and ESM-1b) for the input and yields a leap in accuracy over single-sequence-based techniques in predicting protein 1D secondary and tertiary structural properties, including backbone torsion angles, solvent accessibility and contact numbers for all six test sets (TEST2018, TEST2020, Neff1-2020, CASP12-FM, CASP13-FM and CASP14-FM). More significantly, it has a performance comparable to profile-based methods for those proteins with homologous sequences. For example, the accuracy for three-state secondary structure (SS3) prediction for TEST2018 and TEST2020 proteins are 86.7\% and 79.8\% by SPOT-1D-LM, compared to 74.3\% and 73.4\% by the single-sequence-based method SPOT-1D-Single and 86.2\% and 80.5\% by the profile-based method SPOT-1D, respectively. For proteins without homologous sequences (Neff1-2020) SS3 is 80.41\% by SPOT-1D-LM which is 3.8\% and 8.3\% higher than SPOT-1D-Single and SPOT-1D, respectively. SPOT-1D-LM is expected to be useful for genome-wide analysis given its fast performance. Moreover, high-accuracy prediction of both secondary and tertiary structural properties such as backbone angles and solvent accessibility without sequence alignment suggests that highly accurate prediction of protein structures may be made without homologous sequences, the remaining obstacle in the post AlphaFold2 era.},
	language = {en},
	number = {1},
	urldate = {2023-11-20},
	journal = {Scientific Reports},
	author = {Singh, Jaspreet and Paliwal, Kuldip and Litfin, Thomas and Singh, Jaswinder and Zhou, Yaoqi},
	month = may,
	year = {2022},
	note = {Number: 1
Publisher: Nature Publishing Group},
	keywords = {Machine learning, Protein structure predictions},
	pages = {7607},
	file = {Full Text PDF:/Users/talia_x/Zotero/storage/HTRS5I45/Singh et al. - 2022 - Reaching alignment-profile-based accuracy in predi.pdf:application/pdf},
}

@misc{rao_ESM_1b_2020,
	title = {Transformer protein language models are unsupervised structure learners},
	copyright = {© 2020, Posted by Cold Spring Harbor Laboratory. This pre-print is available under a Creative Commons License (Attribution-NonCommercial-NoDerivs 4.0 International), CC BY-NC-ND 4.0, as described at http://creativecommons.org/licenses/by-nc-nd/4.0/},
	url = {https://www.biorxiv.org/content/10.1101/2020.12.15.422761v1},
	doi = {10.1101/2020.12.15.422761},
	abstract = {Unsupervised contact prediction is central to uncovering physical, structural, and functional constraints for protein structure determination and design. For decades, the predominant approach has been to infer evolutionary constraints from a set of related sequences. In the past year, protein language models have emerged as a potential alternative, but performance has fallen short of state-of-the-art approaches in bioinformatics. In this paper we demonstrate that Transformer attention maps learn contacts from the unsupervised language modeling objective. We find the highest capacity models that have been trained to date already outperform a state-of-the-art unsupervised contact prediction pipeline, suggesting these pipelines can be replaced with a single forward pass of an end-to-end model.1},
	language = {en},
	urldate = {2023-11-20},
	publisher = {bioRxiv},
	author = {Rao, Roshan and Meier, Joshua and Sercu, Tom and Ovchinnikov, Sergey and Rives, Alexander},
	month = dec,
	year = {2020},
	note = {Pages: 2020.12.15.422761
Section: New Results},
	file = {Full Text PDF:/Users/talia_x/Zotero/storage/PN2AZ5W6/Rao et al. - 2020 - Transformer protein language models are unsupervis.pdf:application/pdf},
}

@article{elnaggar_prottrans_2022,
	title = {{ProtTrans}: {Toward} {Understanding} the {Language} of {Life} {Through} {Self}-{Supervised} {Learning}},
	volume = {44},
	issn = {1939-3539},
	shorttitle = {{ProtTrans}},
	url = {https://ieeexplore.ieee.org/document/9477085},
	doi = {10.1109/TPAMI.2021.3095381},
	abstract = {Computational biology and bioinformatics provide vast data gold-mines from protein sequences, ideal for Language Models (LMs) taken from Natural Language Processing (NLP). These LMs reach for new prediction frontiers at low inference costs. Here, we trained two auto-regressive models (Transformer-XL, XLNet) and four auto-encoder models (BERT, Albert, Electra, T5) on data from UniRef and BFD containing up to 393 billion amino acids. The protein LMs (pLMs) were trained on the Summit supercomputer using 5616 GPUs and TPU Pod up-to 1024 cores. Dimensionality reduction revealed that the raw pLM-embeddings from unlabeled data captured some biophysical features of protein sequences. We validated the advantage of using the embeddings as exclusive input for several subsequent tasks: (1) a per-residue (per-token) prediction of protein secondary structure (3-state accuracy Q3=81\%-87\%); (2) per-protein (pooling) predictions of protein sub-cellular location (ten-state accuracy: Q10=81\%) and membrane versus water-soluble (2-state accuracy Q2=91\%). For secondary structure, the most informative embeddings (ProtT5) for the first time outperformed the state-of-the-art without multiple sequence alignments (MSAs) or evolutionary information thereby bypassing expensive database searches. Taken together, the results implied that pLMs learned some of the grammar of the language of life. All our models are available through https://github.com/agemagician/ProtTrans.},
	number = {10},
	urldate = {2023-11-21},
	journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
	author = {Elnaggar, Ahmed and Heinzinger, Michael and Dallago, Christian and Rehawi, Ghalia and Wang, Yu and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and Bhowmik, Debsindhu and Rost, Burkhard},
	month = oct,
	year = {2022},
	note = {Conference Name: IEEE Transactions on Pattern Analysis and Machine Intelligence},
	pages = {7112--7127},
	file = {IEEE Xplore Abstract Record:/Users/talia_x/Zotero/storage/M8EKPSRM/9477085.html:text/html;IEEE Xplore Full Text PDF:/Users/talia_x/Zotero/storage/8NCCTGHB/Elnaggar et al. - 2022 - ProtTrans Toward Understanding the Language of Li.pdf:application/pdf},
}


@article{brandes_proteinBERT_2022,
    author = {Brandes, Nadav and Ofer, Dan and Peleg, Yam and Rappoport, Nadav and Linial, Michal},
    title = "{ProteinBERT: a universal deep-learning model of protein sequence and function}",
    journal = {Bioinformatics},
    volume = {38},
    number = {8},
    pages = {2102-2110},
    year = {2022},
    month = {02},
    abstract = "{Self-supervised deep language modeling has shown unprecedented success across natural language tasks, and has recently been repurposed to biological sequences. However, existing models and pretraining methods are designed and optimized for text analysis. We introduce ProteinBERT, a deep language model specifically designed for proteins. Our pretraining scheme combines language modeling with a novel task of Gene Ontology (GO) annotation prediction. We introduce novel architectural elements that make the model highly efficient and flexible to long sequences. The architecture of ProteinBERT consists of both local and global representations, allowing end-to-end processing of these types of inputs and outputs. ProteinBERT obtains near state-of-the-art performance, and sometimes exceeds it, on multiple benchmarks covering diverse protein properties (including protein structure, post-translational modifications and biophysical attributes), despite using a far smaller and faster model than competing deep-learning methods. Overall, ProteinBERT provides an efficient framework for rapidly training protein predictors, even with limited labeled data.Code and pretrained model weights are available at https://github.com/nadavbra/protein\_bert.Supplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac020},
    url = {https://doi.org/10.1093/bioinformatics/btac020},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/8/2102/45474534/btac020.pdf},
}
```
