# Solubilty-aware AfDesign
### Google Colab
<a href="https://colab.research.google.com/github/blacktanktop/Solubility_AfDesign/blob/main/design.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### setup
```bash
git clone https://github.com/sokrypton/af_backprop.git
pip -q install biopython dm-haiku==0.0.5 ml-collections py3Dmol
mkdir params
curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar | tar x -C params
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/colabfold.py
wget -qnc https://raw.githubusercontent.com/blacktanktop/Solubility_AfDesign/main/design.py
wget -qnc https://raw.githubusercontent.com/blacktanktop/Solubility_AfDesign/solubility/design_util.py
wget -qnc https://raw.githubusercontent.com/blacktanktop/Solubility_AfDesign/solubility/constant.py
```
```python
import numpy as np
from IPython.display import HTML
from design import mk_design_model, clear_mem
import sys
sys.path.append('af_backprop')
```
### binder hallucination
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
