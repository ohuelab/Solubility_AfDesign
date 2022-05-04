import os

from alphafold.common import residue_constants
from alphafold.common import protein

import numpy as np

def get_pdb(pdb_code=""):
  # pdb_dirにpdb fileを保存
  os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
  return f"{pdb_code}.pdb"
  

def to_pdb_binder(prot1, prot2, chain1="A", chain2="B") -> str:
  """Converts a `Protein` instance to a PDB string.
  Args:
    prot: The protein to convert to PDB.
  Returns:
    PDB string.
  """
  restypes = residue_constants.restypes + ['X']
  res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
  atom_types = residue_constants.atom_types
  chain = [chain1, chain2]
  pdb_lines = []
  pdb_lines.append('MODEL     1')
  atom_index = 1
  for idx, prot in enumerate([prot1, prot2]):
    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
  
    if np.any(aatype > residue_constants.restype_num):
      raise ValueError('Invalid aatypes.')
    chain_id = chain[idx]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
      res_name_3 = res_1to3(aatype[i])
      for atom_name, pos, mask, b_factor in zip(
          atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
        if mask < 0.5:
          continue
  
        record_type = 'ATOM'
        name = atom_name if len(atom_name) == 4 else f' {atom_name}'
        alt_loc = ''
        insertion_code = ''
        occupancy = 1.00
        element = atom_name[0]  # Protein supports only C, N, O, S, this works.
        charge = ''
        # PDB is a columnar format, every space matters here!
        atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                     f'{res_name_3:>3} {chain_id:>1}'
                     f'{residue_index[i]:>4}{insertion_code:>1}   '
                     f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                     f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                     f'{element:>2}{charge:>2}')
        pdb_lines.append(atom_line)
        atom_index += 1

  # Close the chain.
  chain_end = 'TER'
  chain_termination_line = (
      f'{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[-1]):>3} '
      f'{chain_id:>1}{residue_index[-1]:>4}')
  pdb_lines.append(chain_termination_line)
  pdb_lines.append('ENDMDL')

  pdb_lines.append('END')
  pdb_lines.append('')
  return '\n'.join(pdb_lines)

def save_binder(model, filename=None, chain1="A", chain2="B"):
  '''save pdb coordinates'''
  outs = model._outs if model._best_outs is None else model._best_outs
  aatype_target = model._batch["aatype"][:model._target_len]
  aatype_binder = outs["seq"].argmax(-1)[0]
  p_t = {"residue_index":model._inputs["residue_index"][0][:model._target_len],
       "aatype":aatype_target,
       "atom_positions":outs["final_atom_positions"][:model._target_len],
       "atom_mask":outs["final_atom_mask"][:model._target_len]}
  p_b = {"residue_index":model._inputs["residue_index"][0][model._target_len:],
       "aatype":aatype_binder,
       "atom_positions":outs["final_atom_positions"][model._target_len:],
       "atom_mask":outs["final_atom_mask"][model._target_len:]}
  b_factors_t = outs["plddt"][:model._target_len:,None] * p_t["atom_mask"]
  b_factors_b = outs["plddt"][model._target_len:,None] * p_b["atom_mask"]
  p_t = protein.Protein(**p_t,b_factors=b_factors_t)
  p_b = protein.Protein(**p_b,b_factors=b_factors_b)
  pdb_lines = to_pdb_binder(p_t, p_b, chain1, chain2)
  if filename is None: return pdb_lines
  else:
    with open(filename, 'w') as f: f.write(pdb_lines)
