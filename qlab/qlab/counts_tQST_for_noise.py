#!/usr/bin/env python3

#
# A general class to provide counts
# extends projectors
#
# This is to be used with maximum-likelihood estimation
#

import itertools as it
import random

import numpy as np
import density_matrix_tool as dmt
import projectors_tQST as PtQST

class Counts_tQST():
    P        = None # Projector class
    projs  = []
    counts = []
    rho      = None # density matrix used for synthetic data
    
    def __init__(self, P):
        self.P = P
            
    def reset_counts(self):
        self.counts   = -np.ones(len(self.proj_names))
        self.proj_idx = []

    def get_counts(self):
        """
        Returns the set of projectors / measurements 
        """
        return self.projs, self.counts
                   
    def set_density_matrix(self, rho):
        self.rho = np.copy(rho)
        
    ######################################################################
    ### Threshold ML tomography
    ######################################################################    

    def get_matrix_elements_tQST(self, threshold, diagonal=None):
        """
        returns a list of tuple of matrix elements for threshold Quantum
        State Tomography:
        in this case we return (i,j) so that |rho_ij| = sqrt(rho_ii rho_jj) >= threshold
        """
        if diagonal is None:
            diagonal = np.diag(self.rho)
            # normalize the trace to 1
            diagonal = diagonal / np.sum(diagonal)
            
        ret = []

        # the diagonal elements
        for i in range(diagonal.size):
            ret.append( (i,i,'r') )

        # the expected density matrix
        exp_rho = np.sqrt(np.outer(diagonal, diagonal))
        M = exp_rho >= threshold        
        # remove the lower triangle (and the diagonal) from the game
        lower_idx = np.tril_indices(len(diagonal))
        M[lower_idx] = False

        # the upper-triangle indices that exceed the threshold
        I,J = np.where( M == True)
        for i,j in zip(I,J):
            ret.append( (i,j,'r') )
            ret.append( (i,j,'i') )

        return(ret)

    def get_matrix_elements_tQST_banded(self, threshold, diagonal=None):
        """
        returns a list of tuple of matrix elements for threshold Quantum
        State Tomography:
        in this case we return the whole band corresponding to elements above threshold on the
        diagonal, that is if rho_ii is above threshold with return (i,j) for all j
        """
        if diagonal is None:
            diagonal = np.diag(self.rho)
            # normalize the trace to 1
            diagonal = diagonal / np.sum(diagonal)

        N = len(diagonal)
        ret = []

        # the diagonal elements
        for i in range(diagonal.size):
            ret.append( (i,i,'r') )

        # indices of the elements above threshold
        idx = [ i for i in range(N) if diagonal[i] >= threshold ]
        # indices of the elements below threshold
        # to be used for including the bands
        off_diagonal = [ i for i in range(N) if diagonal[i] < threshold ]        

        # all the elements above threshold
        above_t = it.product(idx,idx)
        # rows of elements above threshold        
        rows = it.product(idx,off_diagonal)
        # columns of elements above threshold                
        cols = it.product(off_diagonal,idx)
                
        for (i,j) in it.chain(above_t, rows, cols):
            if(j>i):
                ret.append( (i,j,'r') )
                ret.append( (i,j,'i') )
                
        return(ret)
    
    def add_denoising_projectors(self, mel, noise_threshold=0):
        # add the projectors that contribute information to the matrix elements in mel
        # above a given threshold
        # default is to include all projectors
        
        all_projs, all_mel = self.P.all_projectors()

        # find off-diagonal matrix elements we are interested in        
        off_diagonal_mel = [(x[0],x[1]) for x in mel if x[2]=='r' and x[0]!=x[1]]

        # create a list of all the projectors that contribute to the off-diagonal mel
        proj_to_consider = []
        weight = []
        for x in off_diagonal_mel:
            idx = np.bitwise_and(all_projs[:,x[0]]>0 , all_projs[:,x[1]]>0)
            weight = weight + list(np.sum(np.abs(all_projs[idx][:,x])**2,axis=1))
            proj_to_consider = proj_to_consider + [all_mel[i] for i in range(len(idx)) if idx[i]==True]

        # make the list unique
        unique_proj_to_consider = list(set(proj_to_consider))
    
        # new "unique" matrix elements and corresponding weight
        new_mel    = [x for x in unique_proj_to_consider if x not in mel]
        new_weight = np.array([weight[proj_to_consider.index(x)] for x in new_mel])

        # apply the noise threshold
        idx = new_weight > noise_threshold
        mel_to_add = [new_mel[i] for i in range(len(idx)) if idx[i]==True ]

        return mel + mel_to_add , mel_to_add
        
    def get_matrix_elements_tQST_fbk(self, threshold, diagonal=None):
        """
        returns a list of tuple of matrix elements for threshold Quantum
        State Tomography
        """
        if diagonal is None:
            diagonal = np.diag(self.rho)
            # normalize the trace to 1
            diagonal = diagonal / np.sum(diagonal)
            
        ret = []

        # the diagonal elements
        for i in range(diagonal.size):
            ret.append( (i,i,'r') )

        # the expected density matrix (FBK method: eliminates rows and cols)
        diagonal[diagonal < threshold] = 0        
        exp_rho = np.sqrt(np.outer(diagonal, diagonal))
        M = exp_rho >= threshold        
        # remove the lower triangle (and the diagonal) from the game
        lower_idx = np.tril_indices(len(diagonal))
        M[lower_idx] = False

        # the upper-triangle indices that exceed the threshold
        I,J = np.where( M == True)
        for i,j in zip(I,J):
            ret.append( (i,j,'r') )
            ret.append( (i,j,'i') )

        return(ret)

    def get_counts_from_mat_el(self, matrix_element_list):
        """
        Returns a set of projectors / measurements 
        for the given matrix_element_list
        """        
        self.projs = []
        for x in matrix_element_list:
            p = self.P.projector_from_matrix_element(*x)
            self.projs.append( p )

        self.projs = np.array(self.projs)
        p = self.projs.T
        self.counts = np.sum(np.real(np.conj(p) * ( self.rho @ p)), axis=0).flatten()

        return self.projs, self.counts


    def get_counts_from_mat_el_tQST(self, matrix_element_list):
        print("**************************************************************")        
        print("WARNING: get_counts_from_mat_el_tQST is a DEPRECATED FUNCTION!")
        print("Please use get_counts_from_mat_el");
        print("**************************************************************")                
        return(self.get_counts_from_mat_el(matrix_element_list))
    
    def mel_not_measured(self, mel):
        N = self.P.dim
        all_mel = [] 
        for i in range(N):
            all_mel.append( (i,i,'r') ) # probably redundant
            for j in range(i+1,N):
                all_mel.append( (i,j,'r') )
                all_mel.append( (i,j,'i') )
                
        # the matrix element where we assume ZERO counts
        mel_not_measured = [ x for x in all_mel if x not in mel ]
        return mel_not_measured

    def add_zero_counts(self, mel, projs, counts):
        # return ZERO counts on all the projectors not measured in tQST
        mel_zero = self.mel_not_measured(mel)
        # and the corresponding projectors
        projs_zero = [ self.P.projector_from_matrix_element(*x) for x in mel_zero ]
        projs_zero = np.array(projs_zero)

        # update projs
        projs = np.vstack([projs, projs_zero])
        
        # update counts
        counts_zero = np.zeros(len(mel_zero))
        counts = np.hstack([counts, counts_zero]).flatten()

        return projs, counts

    def random_mel_not_measured(self, nrandom, mel):
        mel_not_measured = self.mel_not_measured(mel)
        random_mel = random.sample(mel_not_measured, nrandom)

        return(random_mel)
    
    def add_random_counts(self, nrandom, mel, projs, counts):
        # return ZERO counts on all the projectors not measured in tQST
        random_mel = self.random_mel_not_measured(nrandom, mel)

        new_projs, new_counts = self.get_counts_from_mat_el_tQST(random_mel)

        projs = np.vstack([projs, new_projs])
        counts = np.hstack([counts, new_counts])
        
        return projs, counts
    
    def get_counts_tQST(self, threshold, includeZero=False):
        mel = self.get_matrix_elements_tQST(threshold)
        projs, counts = self.get_counts_from_mat_el_tQST( mel )

        if includeZero == True:
            projs, counts = self.add_zero_counts(mel, projs, counts)

        return projs, counts
    
if __name__ == '__main__':
    nqubit = 4
    
    P2 = PtQST.Projectors_tQST_qubit(nqubit)
    rho = dmt.GHZ(nqubit)

    C2 = Counts_tQST(P2)
    C2.set_density_matrix(rho)

    t = 0.1
    mel = C2.get_matrix_elements_tQST(t)
    print(len(mel),"matrix elements to measure:",mel)

    mel_banded = C2.get_matrix_elements_tQST_banded(t)
    print(len(mel_banded),"matrix elements to measure:",mel_banded)
    
    projs, counts = C2.get_counts_tQST(t)
    print("tQST counts",counts)
    print("shape of projs",projs.shape)
    print("size  of counts",counts.size)    

    nrandom = 4
    print("adding",nrandom,"counts")
    new_mel = C2.random_mel_not_measured(nrandom, mel)
    print("new mel",new_mel)
    print("all mel", mel+new_mel)
    projs, counts = C2.add_random_counts(nrandom,mel, projs, counts)
    print("new counts",counts)
    print("shape of new projs",projs.shape)
    print("size  of new counts",counts.size)    
