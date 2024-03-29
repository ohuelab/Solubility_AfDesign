# Solubilty-aware AfDesign

This repository is based on the AfDesign binder hallucination protocol in ColabDesign, which is published by Dr. Sergey Ovchinnikov on [GitHub](https://github.com/sokrypton/ColabDesign/tree/main/af).


### Google Colab
<a href="https://colab.research.google.com/github/ohuelab/Solubility_AfDesign/blob/solubility/design.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### setup
```bash
git clone https://github.com/sokrypton/af_backprop.git
pip -q install biopython dm-haiku==0.0.5 ml-collections py3Dmol
mkdir params
curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar | tar x -C params
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/colabfold.py
wget -qnc https://raw.githubusercontent.com/ohuelab/Solubility_AfDesign/main/design.py
wget -qnc https://raw.githubusercontent.com/ohuelab/Solubility_AfDesign/solubility/design_util.py
wget -qnc https://raw.githubusercontent.com/ohuelab/Solubility_AfDesign/solubility/constant.py
```
```python
import numpy as np
from IPython.display import HTML
from design import mk_design_model, clear_mem
from design_util import *
import sys
sys.path.append('af_backprop')
```

### About solubility index
Three solubility indices, the Hydrophobicity Index, Hydropathy Index, and Solubility-Weighted Index, were used in this study. 
* The Hydrophobicity Index evaluates hydrophobicity based on the physical characteristics of 20 amino acids to identify regions of a protein's primary sequence that are likely to be buried in the membrane. 
  * Argos, P.; Rao, J.K.M.; Hargrave, P.A. Structural Prediction of Membrane-Bound Proteins. Eur. J. Biochem. 1982, 128, 565-575, doi:10.1111/j.1432-1033.1982.tb07002.x.
* The Hydropathy Index is a hydrophilicity scale that considers the hydrophilicity and hydrophobicity of each of the 20 amino acid side chains and was developed based on experimental observations from the literature. Specifically, values were calculated using both the water vapor transfer free energy and the distribution in and out of the amino acid side chains as determined by Chothia (1976).
  * Kyte, J.; Doolittle, R.F. A Simple Method for Displaying the Hydropathic Character of a Protein. J. Mol. Biol. 1982, 157, 105-132, doi:10.1016/0022-2836(82)90515-0.
* The Solubility-Weighted Index is a predictive index of solubility, and prediction programs using it are superior to many existing de novo protein solubility prediction tools. In this study, the weight of this predictive index was used as the solubility index. 
  * Bhandari, B.K.; Gardner, P.P.; Lim, C.S. Solubility-Weighted Index: Fast and Accurate Prediction of Protein Solubility. Bioinformatics 2020, 36, 4691-4698, doi:10.1093/bioinformatics/btaa578.

### binder hallucination with Solubility index
```python
# solubility_index choose from swi (means Solubility-Weighted Index), hyp (means Hydropathy Index) and hyd (means Hydrophobicity Index)
model = mk_design_model(protocol="binder", solubility_index="swi") 
model.prep_inputs(pdb_filename="1YCR.pdb", chain="A", binder_len=13)
# Specify weights for solubility index. In the paper, we considered between 0~1. However, 0 means that the solubility index is not used.
model.opt["weights"].update({"solubility": 0.5})
model.design_3stage(soft_iters=100, temp_iters=100, hard_iters=10)
```

### binder hallucination (Default)
For a given protein target and protein binder length, generate/hallucinate a protein binder sequence AlphaFold 
thinks will bind to the target structure. To do this, we minimize PAE and maximize number of contacts at the 
interface and within the binder, and we maximize pLDDT of the binder.
```python
model = mk_design_model(protocol="binder")
model.prep_inputs(pdb_filename="1YCR.pdb", chain="A", binder_len=13)
model.design_3stage(soft_iters=100, temp_iters=100, hard_iters=10)
```
# FAQ
#### Can I reuse the same model without needing to recompile?
```python
model.restart()
```
#### What are all the different `design_???` methods?
- For **design** we provide 5 different functions:
  - `design_logits()` - optimize `logits` inputs (continious)
  - `design_soft()` - optimize `softmax(logits)` inputs (probabilities)
  - `design_hard()` - optimize `one_hot(logits)` inputs (discrete)

- For complex topologies, we find directly optimizing one_hot encoded sequence `design_hard()` to be very challenging. 
To get around this problem, we propose optimizing in 2 or 3 stages.
  - `design_2stage()` - `soft` → `hard`
  - `design_3stage()` - `logits` → `soft` → `hard`
#### What are all the different losses being optimized?
- general losses
  - `pae`       - minimizes the predicted alignment error
  - `plddt`     - maximizes the predicted LDDT
  - `msa_ent`   - minimize entropy for MSA design (see example at the end of notebook)
  - `pae` and `plddt` values are between 0 and 1 (where lower is better for both)

- binder specific losses
  - `pae_inter` - minimize PAE interface of the proteins
  - `pae_intra` - minimize PAE within binder
  - `con_inter` - maximize number of contacts at the interface of the proteins
  - `con_intra` - maximize number of contacts within binder

#### How do I change the loss weights?
```python
model.opt["weights"].update({"pae":0.0,"plddt":1.0})
model.opt["weights"]["pae"] = 0.0
```
WARNING: When setting weights be careful to use floats (instead of `1`, use `1.0`), otherwise this triggers recompile.
#### How do I disable or control dropout?
```python
model.opt["dropout_scale"] = 1.0
model.design_???(dropout=True)
```
#### How do I control number of recycles used during design?
```python 
model = mk_design_model(num_recycles=1, recycle_mode="sample")
```
- `num_recycles` - max number of recycles to use during design (for denovo proteins we find 0 is often enough)
- `recycle_model` - When using more than 1 > `num_recycles`:
  - `sample` - at each iteration, randomly select number of recycles to use. (Recommended)
  - `add_prev` - add prediction logits (dgram, pae, plddt) across all recycles. (Most stable, but slow and requires more memory).
  - `last` - only use gradients from last recycle. (NOT recommended).
#### How do I control which model params are used during design?
By default all five models are used during optimization. If `num_models` > 1, then multiple params are evaluated at each iteration 
and the gradients/losses are averaged. Each iteration a random set of model params are used unless `model_mode="fixed"`.
```python
model = mk_design_model(num_models=1, model_mode="sample", model_parallel=False)
```
- `num_models` - number of model params to use at each iteration.
- `model_mode`:
  - `sample` - randomly select models params to use. (Recommended)
  - `fixed` - use the same model params each iteration.
- `model_parallel` - run model params in parallel if `num_models` > 1. By default, the model params are evaluated in serial,
if you have access to high-end GPU, you can run all model params in parallel by enabling this flag. 

#### How is contact defined? How do I change it?
`con` is defined using the distogram, where bins < 20 angstroms are summed. To change the cutoff use:
```python
model.opt["con_cutoff"] = 8.0
```
#### For binder hallucination, can I specify the site I want to bind?
```python
model.prep_inputs(...,hotspot="1-10,15,3")
```
#### How do I set the random seed for reproducibility?
```python
model.restart(seed=0)
```

## References

* Kosugi, T.; Ohue, M. [__Solubility-Aware Protein Binding Peptide Design Using AlphaFold__](https://www.mdpi.com/2227-9059/10/7/1626). _Biomedicines_, 10(7): 1626, 2022. doi: 10.3390/biomedicines10071626

* Kosugi T, Ohue M. [__Solubility-aware protein binding peptide design using AlphaFold__](https://www.biorxiv.org/content/10.1101/2022.05.14.491955). _bioRxiv_, preprint, 2205.02169, 2022. doi: 10.1101/2022.05.14.491955
