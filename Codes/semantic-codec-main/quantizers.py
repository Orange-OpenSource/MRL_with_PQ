"""
/*
* Software Name : CSMRPQ
* SPDX-FileCopyrightText: Copyright (c) Orange SA
* SPDX-License-Identifier: MIT
*
* This software is distributed under the MIT license,
* see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html
*
* Authors:
* Leonardo ROQUE          leonardo.roquealmeidamatos@orange.com
* Louis-Adrien DUFRÈNE    louisadrien.dufrene@orange.com
* Guillaume LARUE         guillaume.larue@orange.com
* Quentin LAMPIN          quentin.lampin@orange.com
*/
"""
from typing import List, Optional, Union
import numpy as np
import torch
import math
from sklearn.cluster import KMeans
from abc import abstractmethod

class Quantizer(torch.nn.Module):

    def __init__(self, output:str) -> None:
        """Quantizer abstract class

        Args:
            output (str): output type, whether 'indices' (index of values) or 'values'. Defaults to 'values'.

        Raises:
            NotImplementedError: this class is meant to be subclassed
    
        """
        super().__init__()
        self.output = output


    @abstractmethod
    def quantize(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """returns a quantized version of x. Depending on the output type, returns indices (integers) or discrete values (float)

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): input to quantize

        Raises:
            NotImplementedError: this method is meant to be overloaded
        Returns:
            torch.Tensor: quantized output
        """
        raise NotImplementedError
    
    @abstractmethod
    def to_indices(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """return the indices of quantized values closest to that of `x`

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): input tensor to quantize

        Raises:
            NotImplementedError: this method is meant to be overloaded

        Returns:
            torch.Tensor: quantized integers
        """
        NotImplementedError
    
    @abstractmethod
    def from_indices(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """returns quantized values corresponding to given indices

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): input indices

        Raises:
            NotImplementedError: this method is meant to be overloaded

        Returns:
            torch.Tensor: quantized values
        """
        raise NotImplementedError


class ProductQuantizer(Quantizer):
    def __init__(self, num_subvecs: int, training_embeds: torch.Tensor, num_clusters: int, output: str) -> None:

        super(ProductQuantizer, self).__init__(output=output)
        self.num_subvecs = num_subvecs
        self.training_embeds = training_embeds
        self.num_clusters = num_clusters
        
    def quantize(self, training_embeds: Union[torch.Tensor, List[torch.Tensor]]) -> Union[np.ndarray, List[np.ndarray]]:
        elements_per_subvec = int(len(training_embeds[0]) / self.num_subvecs) # computes the number of elements composing each subvector
        subvecs_per_embed = [] # list to store the subvectors that compose each original embedding
        #subvec_idx_count = 0 # counter to access the subvectors created for each embedding
        subcodebooks = []
        
        # Splits each embedding into their corresponding subvectors and store them as a list
        for embed in training_embeds:
            subvec_splits_aux = torch.split(embed, elements_per_subvec) # splits the original embedding
            subvec_splits_aux = [np.asarray(split.cpu()) for split in subvec_splits_aux] # converts each subvector created from 'torch.Tensor' to 'numpy.ndarray' format
            subvecs_per_embed.append(subvec_splits_aux) # appends the subvectors to the list
        
        # Groups the sets of subvectors and computes the centroids that represent them. These centroids (per set of matching subvectors) will form the partition's subcodebook
        for subvec_idx in range(len(subvecs_per_embed[0])):
            matching_subvecs = []
            for i in range(len(training_embeds)):
                #matching_subvecs = [subvecs_per_embed[i][subvec_idx] for i in range(len(training_embeds))]
                matching_subvecs.append(subvecs_per_embed[i][subvec_idx]) # groups the matching subvectors in a list
            matching_subvecs = np.concatenate(matching_subvecs, axis=0) # converts the previous list of matching subvectors into a single 'numpy.ndarray' object

            # Sets the kmeans to identify the representative clusters per set of matching subvectors and their respective centroids
            matching_subvecs = matching_subvecs.reshape(-1, elements_per_subvec) # reshapes the matching subvectors before performing kmeans
            kmeans_subvecs = KMeans(n_clusters=self.num_clusters, random_state=0, n_init="auto").fit(matching_subvecs)
            #kmeans_subvecs = KMeans(n_clusters=self.num_clusters).fit(matching_subvecs)

            # Determines the centroids of the identified clusters. These representative vectors are the subcodewords composing the subcodebook for each partition of the embeds
            set_of_subvecs_centroids = kmeans_subvecs.cluster_centers_

            # Appends the computed centroids of each set of subvectors/composes the centroids
            subcodebooks.append(set_of_subvecs_centroids)
        
        return subcodebooks
    
    def to_indices(self, test_embeds: Union[torch.Tensor, List[torch.Tensor]], subcodebooks: Union[np.ndarray, List[np.ndarray]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        elements_per_subvec = int(len(test_embeds[0]) / self.num_subvecs) # computes the number of elements composing each subvector
        subvecs_per_embed = [] # list to store the subvectors that compose each original embedding
        PQ_codes = []
        
        for embed in test_embeds:
            # Splits each embedding into their corresponding subvectors and store them as a list
            subvec_splits_aux = torch.split(embed, elements_per_subvec) # splits the original embedding
            subvec_splits_aux = [np.asarray(split.cpu()) for split in subvec_splits_aux] # converts each subvector created from 'torch.Tensor' to 'numpy.ndarray' format
            subvecs_per_embed.append(subvec_splits_aux) # appends the subvectors to the list

            PQ_code_embed = [] # auxiliar variable to get the indices that will compose the PQ code of each test embedding
            # Computes the PQ codes representing each test embedding
            for subvec_idx in range(len(subvecs_per_embed[0])):
                dists_to_subvec = [ np.linalg.norm(subvec_splits_aux[subvec_idx] - subcodeword)**2 for subcodeword in subcodebooks[subvec_idx] ]
                idx_nearest_neighbor_subvec = np.argmin(dists_to_subvec)
                PQ_code_embed.append(idx_nearest_neighbor_subvec)

            # Appends the entire PQ code of each test embedding
            PQ_codes.append(PQ_code_embed)

        # Converts the generated list into a 'torch.Tensor' object
        PQ_codes = torch.as_tensor(PQ_codes)

        return PQ_codes
    
    def from_indices(self, PQ_codes: Union[torch.Tensor, List[torch.Tensor]], subcodebooks: Union[np.ndarray, List[np.ndarray]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        reconstructed_embeds = [] # list to store the reconstructed embeddings
        elements_per_embed = len(self.training_embeds[0]) # computes the number of elements composing each embedding
        
        # Accessing the PQ code computed for each test embedding, reconstructs all the original embeddings and stores them in a list
        for PQ_code in PQ_codes:
            
            reconstructed_embed = [] # list to store the reconstructed embedding per PQ code

            for subvec_idx in range(len(PQ_codes[0])):
                assigned_subcodeword = subcodebooks[subvec_idx][PQ_code[subvec_idx]]
                reconstructed_embed.append(assigned_subcodeword)
            
            reconstructed_embeds.append(reconstructed_embed)
        
        # Converts the generated list into a 'torch.Tensor' object
        reconstructed_embeds = np.concatenate(reconstructed_embeds, axis=0)
        reconstructed_embeds = reconstructed_embeds.reshape(-1, elements_per_embed)
        reconstructed_embeds = torch.as_tensor(reconstructed_embeds)

        return reconstructed_embeds
