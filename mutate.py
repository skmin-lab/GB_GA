'''
Original code was written by Jan H. Jensen 2018
https://github.com/jensengroup/GB_GA
'''
from rdkit import Chem
from rdkit.Chem import AllChem

import random
import numpy as np
import crossover as co

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def delete_atom(p):
  choices = ['[*:1]~[D1]>>[*:1]', '[*:1]~[D2]~[*:2]>>[*:1]-[*:2]',
             '[*:1]~[D3](~[*;!H0:2])~[*:3]>>[*:1]-[*:2]-[*:3]',
             '[*:1]~[D4](~[*;!H0:2])(~[*;!H0:3])~[*:4]>>[*:1]-[*:2]-[*:3]-[*:4]',
             '[*:1]~[D4](~[*;!H0;!H1:2])(~[*:3])~[*:4]>>[*:1]-[*:2](-[*:3])-[*:4]']
#  p = [0.25,0.25,0.25,0.1875,0.0625]
  nrand = np.random.choice(5, p=p)
  return nrand, choices[nrand]

def append_atom(p):
#  choices = [['single',['C','N','O','F','S','Cl','Br'],7*[1.0/7.0]],
#  choices = [['single',['C','N','O','F','S'],5*[1.0/5.0]],
#             ['double',['C','N','O'],3*[1.0/3.0]],
#             ['triple',['C','N'],2*[1.0/2.0]] ]
#  p_BO = [0.60,0.35,0.05]
#  index = np.random.choice(list(range(3)), p=p_BO)
#  BO, atom_list, p = choices[index]
#  new_atom = np.random.choice(atom_list, p=p)
  choices = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'C', 'N', 'O', 'C', 'N']
  nrand = np.random.choice(12, p=p)
  new_atom = choices[nrand]
  
#  if BO == 'single': 
  if nrand <= 6:
    rxn_smarts = '[*;!H0:1]>>[*:1]X'.replace('X','-'+new_atom)
#  if BO == 'double': 
  elif 6 < nrand <= 9:
    rxn_smarts = '[*;!H0;!H1:1]>>[*:1]X'.replace('X','='+new_atom)
#  if BO == 'triple':
  else: 
    rxn_smarts = '[*;H3:1]>>[*:1]X'.replace('X','#'+new_atom)
    
  return nrand, rxn_smarts

def insert_atom(p):
#  choices = [['single',['C','N','O','F'],4*[1.0/4.0]],
#             ['double',['C','N'],2*[1.0/2.0]],
#             ['triple',['C'],[1.0]]]
#  p_BO = [0.60,0.35,0.05]
#  index = np.random.choice(list(range(3)), p=p_BO)
#  BO, atom_list, p = choices[index]
#  new_atom = np.random.choice(atom_list, p=p)
  
  choices = ['C', 'N', 'O', 'S', 'C', 'N', 'C']
  nrand = np.random.choice(7, p=p)
  new_atom = choices[nrand]
  
#  if BO == 'single': 
  if nrand <= 3:
    rxn_smarts = '[*:1]~[*:2]>>[*:1]X[*:2]'.replace('X',new_atom)
#  if BO == 'double': 
  elif 3 < nrand <= 5:
    rxn_smarts = '[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]'.replace('X',new_atom)
#  if BO == 'triple': 
  else:
    rxn_smarts = '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]'.replace('X',new_atom)
    
  return nrand, rxn_smarts

def change_bond_order(p):
  choices = ['[*:1]!-[*:2]>>[*:1]-[*:2]','[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]',
             '[*:1]#[*:2]>>[*:1]=[*:2]','[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]']
#  p = [0.45,0.45,0.05,0.05]

  nrand = np.random.choice(4, p=p)
  return nrand, choices[nrand]

def delete_cyclic_bond():
  return 0, '[*:1]@[*:2]>>([*:1].[*:2])'

def add_ring(p):
  choices = ['[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1',
             '[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1',
             '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1',
             '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1'] 
#  p = [0.05,0.05,0.45,0.45]
  nrand = np.random.choice(4, p=p)
  return nrand, choices[nrand]

def change_atom(mol, p):
  choices = ['#6','#7','#8','#9','#16','#17','#35']
#  select_p = [0.15,0.15,0.14,0.14,0.14,0.14,0.14]
#  choices = ['#6','#7','#8','#9','#16']
#  p = [0.2,0.2,0.2,0.2,0.2]
  
  X = np.random.choice(choices, p=p)
  while not mol.HasSubstructMatch(Chem.MolFromSmarts('['+X+']')):
    X = np.random.choice(choices, p=p)
  
  nrand = np.random.choice(7, p=p)
  Y = choices[nrand]
  while Y == X:
    nrand = np.random.choice(7, p=p)
    Y = choices[nrand]
  
  return nrand, '[X:1]>>[Y:1]'.replace('X',X).replace('Y',Y)

def mutate(mol,p):
  Chem.Kekulize(mol,clearAromaticFlags=True)
#  p = [0.15,0.14,0.14,0.14,0.14,0.14,0.15]
  for i in range(10):
    rxn_smarts_num = np.random.choice([0,1,2,3,4,5,6], p=[sum(p[pnum]) for pnum in range(7)])

    if rxn_smarts_num == 0:
      rxn_detailed_num, rxn_smarts = insert_atom(p[0]/sum(p[0]))
    elif rxn_smarts_num == 1:
      rxn_detailed_num, rxn_smarts = change_bond_order(p[1]/sum(p[1]))
    elif rxn_smarts_num == 2:
      rxn_detailed_num, rxn_smarts = delete_cyclic_bond()
    elif rxn_smarts_num == 3:
      rxn_detailed_num, rxn_smarts = add_ring(p[3]/sum(p[3]))
    elif rxn_smarts_num == 4:
      rxn_detailed_num, rxn_smarts = delete_atom(p[4]/sum(p[4]))
    elif rxn_smarts_num == 5:
      rxn_detailed_num, rxn_smarts = change_atom(mol, p[5]/sum(p[5]))
    elif rxn_smarts_num == 6:
      rxn_detailed_num, rxn_smarts = append_atom(p[6]/sum(p[6]))
    else:
      raise IndexError(f"Invalid mutation #: {rxn_smarts_num}")  

#    rxn_smarts_list = 7*['']
#    rxn_detailed_num, rxn_smarts_list[0] = insert_atom(p[0]/sum(p[0]))
#    rxn_detailed_num, rxn_smarts_list[1] = change_bond_order(p[1]/sum(p[1]))
#    rxn_detailed_num, rxn_smarts_list[2] = delete_cyclic_bond()
#    rxn_detailed_num, rxn_smarts_list[3] = add_ring(p[3]/sum(p[3]))
#    rxn_detailed_num, rxn_smarts_list[4] = delete_atom(p[4]/sum(p[4]))
#    rxn_detailed_num, rxn_smarts_list[5] = change_atom(mol, p[5]/sum(p[5]))
#    rxn_detailed_num, rxn_smarts_list[6] = append_atom(p[6]/sum(p[6]))
#    rxn_smarts = np.random.choice(rxn_smarts_list, p=p) 
#    rxn_smarts = rxn_smarts_list[rxn_smarts_num] 
    
#    print('mutation',rxn_smarts_num)
    
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)

    new_mol_trial = rxn.RunReactants((mol,))

    new_mols = []
    for m in new_mol_trial:
      m = m[0]
#      print(Chem.MolToSmiles(mol),mol_OK(mol))
#      print(Chem.MolToSmiles(m))
      if co.mol_OK(m) and co.ring_OK(m):
        new_mols.append(m)
    if len(new_mols) > 0:
      return rxn_smarts_num, rxn_detailed_num, random.choice(new_mols)
  
  return None, None, None

if __name__ == "__main__":
    co.average_size = 39.15
    co.size_stdev = 23.50
    mutation_rate = 1.0
    co.string_type = 'SMILES'

    string = 'CCC(CCCC)C'
#    string = 'c1cc2ccccc2cc1'
    string = Chem.MolFromSmiles(string)
    for i in range(20):
      child = mutate(string,mutation_rate)
      print(Chem.MolToSmiles(child))
