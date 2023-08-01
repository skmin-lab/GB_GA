#!/usr/bin/env python

#################################################################################################
# Import module
#################################################################################################

import sys
import os, shutil
sys.path.append('..')  # To import from GP.kernels and property_predition.data_utils
import random

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import crossover as co
import mutate as mu
import math

import sascorer as sascorer
from rdkit import Chem
from rdkit.Chem import AllChem

from GP.kernels import Tanimoto
from property_prediction.data_utils import transform_data, TaskDataLoader, featurise_mols

#GPU designation
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

#################################################################################################
# Main function
#################################################################################################

def main():

    # Load learning sets
    data_loader = TaskDataLoader('QM_E1_CAM', 'dluc_analog.csv')
    smiles_list, y1 = data_loader.load_property_data()
    X = featurise_mols(smiles_list, 'fingerprints')
    
    X = X.astype(np.float64)
    y = y1.reshape(-1, 1)

    model, y_scaler = GPRmodelgen(X, y)

    ############################
    ### parameters for GB-GA ###
    # GA parameters
    nstep = 100
    target_pool = 1000
    target_value = 0.041421
    
    # Set range of number of heavy atoms in molecule
    co.average_size = 14
    co.size_stdev = 5
    co.string_type = 'SMILES'
    
    # Gaussian noise range for selection
    gau_sigma = 0.001
    ############################
    
    ####################################
    ### Initial mutation probability ###
    # Make individual probability array for each mutation group, and then merge all
    # In the design of D-luc analogues, some mutations are forced to be forbidden
    p_part = [0.5, 0.075, 0.07, 0.07, 0.07, 0.07, 0.07, 0.075]
    p_individual = []
    p_individual.append([0.175, 0.175, 0.175, 0., 0.2, 0.2, 0.075])       # p_insert_atom(C1,N1,O1,F1,C2,N2,C3)
    p_individual.append([0.45, 0.45, 0.05, 0.05])                         # p_change_bond_order 
    p_individual.append([1.])                                             # p_delete_cyclic_bond
    p_individual.append([0.05, 0.05, 0.45, 0.45])                         # p_add_ring
    p_individual.append([0.25, 0.25, 0.25, 0.1875, 0.0625])               # p_delete_atom
    p_individual.append([0.25, 0.25, 0.25, 0. ,0.25, 0., 0.])             # p_change_atom
    p_individual.append([0.1505, 0.1505, 0.1505, 0., 0.1505, 0., 0., 0.116, 0.116, 0.116, 0.025, 0.025]) # p_append_atom
    mu_prob = [p_part[0]]
    
    for nprob in range(7):
        append_array = [elements * p_part[nprob + 1] for elements in p_individual[nprob]]
        mu_prob.append(append_array)  
    ####################################

    # Initial genetic pool generation
    GenPool = []
    for elements in smiles_list:
        GenPool.append(elements)
 
    with open(f"log_mutation", "w") as f: # Initialize mutation log
        f.write("")

    print("#cycle  cut/mean/std  total_pool(from selection+from crossover&mutation)  number_of_crossover  number_of_mutation  number_of_repetition  SAS_mean/std", flush=True)
    for istep in range(nstep):
        mu_prob, GenPool = GB_GA(mu_prob, model, GenPool, istep, y_scaler, target_value, gau_sigma, target_pool)

#################################################################################################
# Calculation functions
#################################################################################################

# Model generation from FlowMO example code
def GPRmodelgen(X, y):
    # Function to minimize in prior optimization
    m1 = None
    def objective_closure():
        return -m1.log_marginal_likelihood()
    
    # Separate test & train set in given data
    # all molecules in the db are used in training(by minimalize test size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    
    # Select kernel and generate model object
    k = Tanimoto()
    m1 = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=k, noise_variance=1)
    
    # Optimize with Scipy
    opt = gpflow.optimizers.Scipy()
    opt.minimize(objective_closure, m1.trainable_variables, options=dict(maxiter=1000))

#    Model evaluation    
#    y_pred, y_var = m1.predict_f(X_test)
#    y_pred = y_scaler.inverse_transform(y_pred)
#    y_test = y_scaler.inverse_transform(y_test)
    
     # Compute R^2, RMSE and MAE on test set molecules
#    score = r2_score(y_test, y_pred)
#    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#    mae = mean_absolute_error(y_test, y_pred)
#    
#    print("\nR^2: {:.3f}".format(score))
#    print("RMSE: {:.3f}".format(rmse))
#    print("MAE: {:.3f}".format(mae))

    return m1, y_scaler

def GB_GA(mu_prob, model, GenPool, istep, y_scaler, target_value, gau_sigma, target_pool):
    # Arrays for mutation evalutation
    mu_eval = [0, [0] * 7, [0] * 4, [0] * 1, [0] * 4, [0] * 5, [0] * 7, [0] * 12]
    mu_eval_count = [0, [0] * 7, [0] * 4, [0] * 1, [0] * 4, [0] * 5, [0] * 7, [0] * 12]

    # Convert molecules included in the pool to fingerprint 
    GenPool_fp = featurise_mols(GenPool, 'fingerprints')
    GenPool_fp = GenPool_fp.astype(np.float64)

    # Listing SAS of individual molecule
    sas_list = []
    for smi in GenPool:
        mol = Chem.MolFromSmiles(smi)
        sas_list.append(sascorer.calculateScore(mol)) # SAscore in RDKit
    sas_list = np.array(sas_list, dtype=float)
    sas_mean = np.mean(sas_list)
    sas_std = np.std(sas_list)
  
    # Predict molecular properties from a GP model
    y_pred, y1_var = model.predict_f(GenPool_fp)
    y_pred = y_scaler.inverse_transform(y_pred)

    # Set criteria for cutting limit of fitness value 
    sort_a = []  # array for cutting criteria
    score = []

    for imol in range(len(y_pred)):
        gau_target = np.random.normal(target_value, gau_sigma) # Apply gaussian noise in the generation model
        diff = abs(gau_target - y_pred[imol][0])
#        score =  (1 / diff) * math.exp(-0.5 * ((max(sas_list[imol], SA_mu) - SA_mu) / SA_sigma) ** 2)  # using heu score
    
        # Cutting based on score, output depends on diff
        score.append(diff)
        sort_a.append(diff)

    sort_a.sort()
    cut = sort_a[int(len(sort_a) * 0.2)] # Cutoff value for selection

    X_parents1 = []  # Array for parents molecules, obtain from selection
    X_parents2 = []  # Array for parents molecules, obtain from crossover & mutation
    mol_list = [] # Array for statistical analysis of high-score molecules
  
    #### Selection ####
    if (istep == 0): # Pass selection step in the first step
        X_parents1 = GenPool
    else:
        over_cut = f"High-score molecules based on the cutoff : {cut:9.5f} (a.u.)\n" # Molecules which fitness is over than cutoff
        less_cut = f"\nLow-score molecules based on the cutoff : {cut:9.5f} (a.u.)\n" # Molecules which fitness is less than cutoff
        for imol in range(len(GenPool)):
            mol = Chem.MolFromSmiles(GenPool[imol])
            if (sascorer.calculateScore(mol) > 3.5): # Reject if molecule has bad sascore
                continue
            rand = random.random()
            diff = abs(target_value - y_pred[imol][0])
            if (score[imol] < cut): # Selection of high-ranked molecule
                X_parents1.append(GenPool[imol])
                over_cut += f"{imol:8.0f} th molecule: {GenPool[imol]} , {y_pred[imol][0]:9.5f} (a.u.), {sas_list[imol]:7.2f}\n"
                mol_list.append(y_pred[imol][0])
            elif (score[imol] >= cut): # Selection of low-ranked molecule
                less_cut += f"{imol:8.0f} th molecule: {GenPool[imol]} , {y_pred[imol][0]:9.5f} (a.u.), {sas_list[imol]:7.2f}\n"
                if (rand >= 0.8): # Selection depends on random number
                    X_parents1.append(GenPool[imol]) 

        # Logging step index
        with open(f"steps/INDEX_{istep}.dat", "w") as f:
            f.write(over_cut)
            f.write(less_cut)

    mean = np.mean(mol_list)
    std = np.std(mol_list)

    #### Crossover & Mutation ####
    ncross = 0 # Number of crossover, for logging
    nmut = 0 # Number of mutation, for logging
    need_offspring = target_pool - len(X_parents1) # Number of molecules to be crossover & mutation, required for sustain pool size
    gau_sigma_param = 0.5 # Parameter for mutation

    # Generate molecules until the pool reaches required pool size
    while (ncross < need_offspring):
           
        X_tmp1 = random.choice(X_parents1)
        X_tmp2 = random.choice(X_parents1)

        mol1 = Chem.MolFromSmiles(X_tmp1)
        mol2 = Chem.MolFromSmiles(X_tmp2)

        child = co.crossover(mol1,mol2)

        if (child == None):
            continue # Exceptional case for return 'None'

        l_mut = False # Logical to check mutation
        if (random.random() > mu_prob[0]):
            l_mut = True

            sum_prob = np.sum([sum(elements) for elements in mu_prob[1:]])
            mu_prob_fix = []
            for elements in mu_prob[1:]:
                mu_prob_fix.append(elements/sum_prob) # Normalize mutation probabilty

            mu_num, detailed_num, mutated_child = mu.mutate(child, mu_prob_fix)

            if (mutated_child == None):
                continue # Exceptional case for return 'None'
  
            eval_mutation = [child, mutated_child] # Array for evalutate mutation
            child = mutated_child

        cano_child = Chem.MolToSmiles(child, True)
        # If a molecule does not have thio- group, reject it
        if (featurise_mols([cano_child], 'fingerprints')[0][719] != 1 or featurise_mols([cano_child], 'fingerprints')[0][1216] != 1):
            continue
        # If a molecule does not have required sascore, reject it
        if (sascorer.calculateScore(child) > 3.5):
            continue
        # If a molecule is already included in the pool, reject it
        if cano_child in X_parents2:
            continue

        ncross += 1

        # Routine for counting mutation numbers
        if (l_mut == True):
            nmut += 1
            mut_fp = featurise_mols([Chem.MolToSmiles(eval_mutation[0]), Chem.MolToSmiles(eval_mutation[1])], 'fingerprints')
            mut_fp = mut_fp.astype(np.float64)

            eval_mutation_value, mut_var = model.predict_f(mut_fp)

#            # mutation logging for heu score
#            score_1 =  (1 / abs(target_value - eval_mutation_value[0])) * math.exp(-0.5 * ((max(sascorer.calculateScore(eval_mutation[0]), SA_mu) - SA_mu) / SA_sigma) ** 2)
#            score_2 =  (1 / abs(target_value - eval_mutation_value[1])) * math.exp(-0.5 * ((max(sascorer.calculateScore(eval_mutation[1]), SA_mu) - SA_mu) / SA_sigma) ** 2)

            score_1 =  abs(target_value - eval_mutation_value[0]) 
            score_2 =  abs(target_value - eval_mutation_value[1]) 

            # Record number of progressive and regressive mutations
            if (score_1 > score_2):
                mu_eval[mu_num + 1][detailed_num] -= 1 
                mu_eval_count[mu_num + 1][detailed_num] += 1
            elif (score_1 <= score_2):
                mu_eval[mu_num + 1][detailed_num] += 1
                mu_eval_count[mu_num + 1][detailed_num] += 1

        X_parents2.append(cano_child)

    #### Mutation control step ####
    norm_eval = [0, [0] * 7, [0] * 4, [0] * 1, [0] * 4, [0] * 5, [0] * 7, [0] * 12]
    eval_count = 0
    for elements in mu_eval_count[1:]: # do not consider mutation probability(mu_eval_count[0])
        eval_count += sum(elements)

    for nmut_1 in range(1, len(mu_eval)):
        for nmut_2 in range(len(mu_eval[nmut_1])):
            norm_eval[nmut_1][nmut_2] = mu_eval[nmut_1][nmut_2] / eval_count # normalize probability

    log_mut = ""
    log_mut += f"----------step{istep}------------\n"
    log_mut += f"counting for mutation evaluation: {mu_eval}\n"
    log_mut += f"normalized: {norm_eval}\n"
    log_mut += f"mu. count(total {eval_count}): " + f"{mu_eval_count}\n"
    for nmut_1 in range(1, len(mu_eval)):
        log_mut += "|--------------------------------------|\n"
        for nmut_2 in range(len(mu_eval[nmut_1])):
            gau_random = abs(np.random.normal(0, gau_sigma_param * abs(norm_eval[nmut_1][nmut_2]) * mu_prob[nmut_1][nmut_2]))
            if mu_eval[nmut_1][nmut_2] > 0:
                mu_prob[nmut_1][nmut_2] += gau_random
                log_mut += f"| {nmut_1:3.0f}  {nmut_2:3.0f}   +{gau_random:12.6f} {mu_prob[nmut_1][nmut_2]:12.6f}|\n"
            else:
                mu_prob[nmut_1][nmut_2] -= gau_random
                log_mut += f"| {nmut_1:3.0f}  {nmut_2:3.0f}   -{gau_random:12.6f} {mu_prob[nmut_1][nmut_2]:12.6f}|\n"
    log_mut += "|--------------------------------------|\n"

    # Mutation probablity normalize when the probability of mutation fluctuates
    sum_mu_prob = 0
    for elements in mu_prob[1:]:
        sum_mu_prob += sum(elements)
    for nmut_1 in range(1, len(mu_prob)):
        for nmut_2 in range(len(mu_prob[nmut_1])):
            mu_prob[nmut_1][nmut_2] /= (1 / (1 - mu_prob[0])) * sum_mu_prob

    # np.set_printoptions(suppress=True)
    log_mut += f"updated array={mu_prob}\n\n"

    with open(f"log_mutation", "a") as f:
        f.write(log_mut)

    # Convert to canonical smiles and remove reapeated elements
    GenPool = np.concatenate([X_parents1, X_parents2])
    GenPool_canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),True) for smi in GenPool]

    GenPool_remove_rep = []
    for elements in GenPool_canonical:
      if elements not in GenPool_remove_rep:
        GenPool_remove_rep.append(elements)

    repetition = len(GenPool)-len(GenPool_remove_rep)
    GenPool = GenPool_remove_rep

    # Stdout in istep
    if (istep != 0):
      print(f"{(istep+1):6.0f}  {cut:10.5f} / {mean:8.5f} / {std:8.5f}  {len(GenPool):6.0f} ({len(X_parents1):6.0f} + {len(X_parents2):6.0f}) {ncross:6.0f} {nmut:6.0f} {repetition:6.0f} {sas_mean:8.2f} / {sas_std:8.2f}", flush=True)

    return mu_prob, GenPool

#################################################################################################
# Why python using this? :P
#################################################################################################

if __name__ == "__main__":
    # Initialize output directory
    path = "./steps"
    if (os.path.exists(path)):
        if (os.path.exists(path + "_old")):
            shutil.rmtree(path + "_old")
        shutil.move(path, path + "_old")
    os.makedirs(path)

    main()

