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
###### Importing the necessary libraries ######
import argparse
import logging
import os
import re
import torch

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from typing import List, Optional, Union

from mteb import MTEB
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from binarizers import Binarizer, IEEE754Binary32Binarizer, NaiveBinarizer
from channels import Channel, IdentityChannel
from quantizers import Quantizer, ProductQuantizer
from mteb_local import SST2BinarySentClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"


###### Defining some helpful variables ######
channel_name = "IdentityChannel"
quantizer_name = None
dataset_name = "banking77" # possible options: "banking77", "stanfordnlp/sst2"
matryoshka_full_dim = 768
matryoshka_trunc_dim = 1 # sets the number of dimensions that embeddings will be truncated with ([1,2,4,8,16,32,64,128,256,512,768])
vec_n_subvecs = [1] # vector specifying the multiple numbers of subvectors to be created during the Product Quantization process


###### Setting up a reference model which computes the embeddings ######
class ReferenceModel(torch.nn.Module):
    def __init__(self, model, device=None, matryoshka_trunc_dim=None):
        super(ReferenceModel, self).__init__()
        
        self.model = model
        self.model = model.eval()

        self.device = device
        if self.device == 'multi':
            self.pool = self.model.start_multi_process_pool()
        elif self.device != None:
            self.model.to(device)

        self.matryoshka_trunc_dim = matryoshka_trunc_dim
     
    def encode(self,
        sentences: Union[str, List[str]],
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
        task_name: Optional[str] = None,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """Encodes the input text sentences to their corresponding embeddings representation

        Args:
            sentences (str list): string list containing all the text sentences
            prompt_name
            prompt
            batch_size
            show_progress_bar
            output_value
            convert_to_numpy
            convert_to_tensor
            device
            normalize_embeddings

        Returns:
            torch.Tensor: embeddings
        """
        with torch.no_grad():
            if self.device == 'multi':
                logging.info('encoding sentences using multi_process')
                embeddings = self.model.encode_multi_process(
                    sentences=sentences,
                    pool=self.pool,
                    prompt_name=prompt_name,
                    prompt=prompt,
                    batch_size=batch_size,
                    chunk_size=None,
                    normalize_embeddings=normalize_embeddings
                )
            else:
                embeddings = self.model.encode(
                    sentences,
                    prompt_name,
                    prompt,
                    batch_size,
                    show_progress_bar,
                    output_value,
                    convert_to_numpy=False,
                    convert_to_tensor=convert_to_tensor,
                    device=device,
                    normalize_embeddings=normalize_embeddings
                )

        embeddings = [emb.cpu() for emb in embeddings]

        # As Base model (with Sentence Transformers library) does not have the parameter to truncate the generated embeds,
        # we first create the full-dimensioned embeds and then truncate manually until the desired dimension
        trunc_embeds_as_list = []
        for embed in embeddings:
            embed = embed[0:self.matryoshka_trunc_dim] # gets only the 'matryoshka_trunc_dim' first elements of each embed
            trunc_embeds_as_list.append(embed)
        embeddings = np.array(trunc_embeds_as_list)
        embeddings = torch.as_tensor(embeddings)
        
        if convert_to_numpy is True:
            embeddings = np.asarray([emb.numpy() for emb in embeddings])
        
        # Implements a DATA SCALING to ensure the classifier (especially 'LogisticRegression') will achieve convergence
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

        # Replaces NaN values by 0 and infinity values by very large finity values
        embeddings = np.nan_to_num(embeddings)

        # Converts the embeddings into tensor objects
        embeddings = torch.as_tensor(embeddings)

        return embeddings
    

###### Setting up the main processing class which defines the whole study scheme ######
class StudyModel(torch.nn.Module):
    def __init__(self, model, dataset_name, quantizer_name, subcodebooks: Optional[Union[np.ndarray, List[np.ndarray]]], quantizer:Quantizer, binarizer: Binarizer, channel:Channel, device:str):
        """_summary_

        Args:
            model (torch.nn.Module): sentence embeddings model
            dataset_name (str): name of the considered dataset
            quantizer (Quantizer): quantizer
            binarizer (Binarizer): binarizer
            channel (Channel): channel
            device (str): device to perform a processing task 

        """
        super(StudyModel, self).__init__()
        
        self.model = model

        self.dataset_name = dataset_name
        self.quantizer_name = quantizer_name

        self.quantizer = quantizer
        if self.quantizer is not None:
            self.activate_quantization = True
            if self.quantizer_name=="ProductQuantizer":
                self.subcodebooks = subcodebooks
            else:
                self.subcodebooks = None
        else:
            self.activate_quantization = False

        self.binarizer = binarizer
        self.channel = channel

    def encode(self,
        sentences: Union[str, List[str]],
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
        task_name: Optional[str] = None,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        
        # Computes the embedding representations of text sentences
        embeddings = self.model.encode(
            sentences,
            prompt_name,
            prompt,
            batch_size,
            show_progress_bar,
            output_value,
            convert_to_numpy=False,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings
        )

        if isinstance(embeddings, list) and self.dataset_name=="banking77" and self.activate_quantization==False:
            # logging.info('binarizing embeddings')
            binarized_embeddings = [self.binarizer.encode(embedding) for embedding in embeddings]
            # logging.info('channeling the binary versions of embeddings')
            rx_binarized_embeddings = [self.channel.apply(binarized_embedding) for binarized_embedding in binarized_embeddings]
            # logging.info('decoding received binarized embeddings')
            rx_embeddings = [self.binarizer.decode(rx_binarized_embedding) for rx_binarized_embedding in rx_binarized_embeddings]
            # Sets up the right output for returning the method
            rx_quantized_embeddings = rx_embeddings # as we are NOT performing quantization for this case

        elif isinstance(embeddings, torch.Tensor) and self.dataset_name=="banking77" and self.activate_quantization==False:
            # logging.info('binarizing embeddings')
            binarized_embeddings = self.binarizer.encode(embeddings)
            # logging.info('channeling the binary versions of embeddings')
            rx_binarized_embeddings = self.channel.apply(binarized_embeddings)
            # logging.info('decoding received binarized embeddings')
            rx_embeddings = self.binarizer.decode(rx_binarized_embeddings)
            # Sets up the right output for returning the method
            rx_quantized_embeddings = rx_embeddings # as we are NOT performing quantization for this case
        else:
            if self.quantizer_name=="ProductQuantizer":
                # logging.info('quantizing embeddings #3')
                #embeddings_tensor: torch.Tensor = torch.Tensor(embeddings)
                embeddings_tensor = embeddings
                quantized_embeddings = self.quantizer.to_indices(embeddings_tensor, self.subcodebooks)
                # logging.info('binarizing embeddings #3')
                binarized_embeddings = self.binarizer.encode(quantized_embeddings)
                # logging.info('channeling indices #3')
                rx_embeddings = self.channel.apply(binarized_embeddings)
                # logging.info('decoding received indices #3')
                rx_indices = self.binarizer.decode(rx_embeddings)
                # logging.info('reconstructing received embeddings #3')
                rx_quantized_embeddings = self.quantizer.from_indices(rx_indices, self.subcodebooks)
            else:
                # logging.info('binarizing embeddings')
                binarized_embeddings = self.binarizer.encode(embeddings)
                # logging.info('channeling the binary versions of embeddings')
                rx_binarized_embeddings = self.channel.apply(binarized_embeddings)
                # logging.info('decoding received binarized embeddings')
                rx_embeddings = self.binarizer.decode(rx_binarized_embeddings)
                # Sets up the right output for returning the method
                rx_quantized_embeddings = rx_embeddings # as we are NOT performing quantization for this case
        
        if convert_to_numpy is True:
            rx_quantized_embeddings = np.asarray([emb.cpu().numpy() for emb in rx_quantized_embeddings])
        
        # Implements a DATA SCALING to ensure the classifier (specially 'LogisticRegression') will achieve convergence
        scaler = StandardScaler()
        rx_quantized_embeddings = scaler.fit_transform(rx_quantized_embeddings)

        # Replaces NaN values by 0 and infinity values by very large finity values
        rx_quantized_embeddings = np.nan_to_num(rx_quantized_embeddings)

        return rx_quantized_embeddings


###### Defining some helpful functions ######
def make_safe_path_name(name):
    # Replace or remove invalid characters
    safe_name = re.sub(r'[\\/*?:"<>|]', '-', name)
    safe_name = re.sub(r'[\s]+', '_', safe_name)
    safe_name = safe_name.strip().lower()
    return safe_name


###### Defining the main executable ######
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model used to generate sentence embeddings *(tomaarsen/mpnet-base-nli-matryoshka)", default="tomaarsen/mpnet-base-nli-matryoshka", type=str)
    parser.add_argument("--output-folder", help="output folder for benchmark results", default="./benchmark-results/tomaarsen-mpnet-base-nli-matryoshka/banking77/Truncated")
    parser.add_argument("--verbose", action='store_true', help="verbose mode", default=True)
    args = parser.parse_args()

    model_name:str = args.model
    output_folder_path: str = args.output_folder
    verbose: bool = args.verbose

    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)
    

    # Checking if GPU processing is available and setting the current device
    torch.set_printoptions(threshold=1000)
    torch.set_printoptions(sci_mode=True, precision=3)
 
    if torch.backends.mps.is_available():
        current_device = torch.device('mps')
        print("MPS is available")
    elif torch.cuda.is_available():
        current_device = torch.device('cuda:0')
        print(f"CUDA is available")

        num_gpus = torch.cuda.device_count()
        current_device_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device_index)
        print(f"Number of CUDA-GPUs: {num_gpus}")
        print(f"Current GPU device index: {current_device_index}")
        print(f"GPU name: {gpu_name}")
    else:
        current_device = torch.device('cpu')
        print("CPU is available")
    
    print(f'Current device is {current_device}')


    # Loading the reference model and defining the target task(s)
    reference_model: ReferenceModel = ReferenceModel(model=SentenceTransformer(model_name, device=current_device, trust_remote_code=True), device=current_device, matryoshka_trunc_dim=matryoshka_trunc_dim)
    #tasks: List[str] = ["SST2BinarySentClassification"]
    tasks: List[str] = ["Banking77Classification"]
    # If SST2 binary sentiment classification is performed, sets up the corresponding task class implemented in the folder "mteb_local". The refered task is not defined in mteb library
    for task in tasks:
        if task=="SST2BinarySentClassification":
            task = [SST2BinarySentClassification()]


    # Getting the training embeddings as a 'torch.Tensor' object to perform the quantization process
    #file_path_train_embeds = f"./embeddings/tomaarsen-mpnet-base-nli/banking-77_tomaarsen-mpnet-base-nli_train-embeds_not-normalized_{base_trunc_dim}_dims.csv"
    #df_train_embeds = pd.read_csv(file_path_train_embeds)
    #
    #training_embeds = []
    #for i in range( len(df_train_embeds.iloc[0, 1:]) ):
    #    train_embed_as_df = df_train_embeds.iloc[:, i+1]
    #    training_embeds.append(train_embed_as_df)
    #
    #scaler = StandardScaler()
    #training_embeds = scaler.fit_transform(training_embeds)
    #
    #training_embeds = torch.FloatTensor(training_embeds)


    # full-resolution embeddings
    evaluation = MTEB(tasks=tasks)
    model_name = make_safe_path_name(model_name)
    results_path = os.path.join(output_folder_path, f'{model_name}_{matryoshka_trunc_dim}_dims', 'original')
    evaluation.run(reference_model, overwrite_results=True, output_folder=results_path)

    quantizer = None
    subcodebooks = None
    
    binarizer = IEEE754Binary32Binarizer()
    scores = {task:[] for task in tasks}

    if quantizer_name=="ProductQuantizer":
        for n_subvecs in vec_n_subvecs:
            training_embeds = None # as we are NOT performing quantization in this case
            quantizer = ProductQuantizer(num_subvecs=n_subvecs, training_embeds=training_embeds, num_clusters=2, output='embeddings')
            subcodebooks = quantizer.quantize(training_embeds=training_embeds)
            logging.info(f"evaluating model on '{channel_name}' and using '{quantizer_name}' with 'n_subvecs' = {n_subvecs} as quantization")
            channel = IdentityChannel()
            study_model = StudyModel(model=reference_model, dataset_name=dataset_name, quantizer_name=quantizer_name, subcodebooks=subcodebooks, quantizer=quantizer, binarizer=binarizer, channel=channel, device=current_device)
            evaluation = MTEB(tasks=tasks)
            model_name = make_safe_path_name(model_name)
            results_path = os.path.join(output_folder_path, f'{model_name}_{matryoshka_trunc_dim}_{matryoshka_full_dim}_dims', 'PQ_NaiveBinarizer', f'IdentityChannel', f'{n_subvecs}')
            results = evaluation.run(study_model, reference_model=reference_model, overwrite_results=True, output_folder=results_path)
            logging.info(f"results: {results}")
            results = results[0].to_dict()
            for task in tasks:
                scores[task].append(results['scores']['test'][0]['main_score'])
        
        results_df = pd.DataFrame(scores, index=vec_n_subvecs)
    else:
        logging.info(f"evaluating model on '{channel_name}' and using '{matryoshka_trunc_dim}' as the truncated dimension")
        channel = IdentityChannel()
        study_model = StudyModel(model=reference_model, dataset_name=dataset_name, quantizer_name=quantizer_name, subcodebooks=subcodebooks, quantizer=quantizer, binarizer=binarizer, channel=channel, device=current_device)
        evaluation = MTEB(tasks=tasks)
        model_name = make_safe_path_name(model_name)
        results_path = os.path.join(output_folder_path, f'{model_name}_{matryoshka_trunc_dim}_dims', 'Truncated', f'IdentityChannel')
        results = evaluation.run(study_model, reference_model=reference_model, overwrite_results=True, output_folder=results_path)
        logging.info(f"results: {results}")
        results = results[0].to_dict()
        for task in tasks:
            scores[task].append(results['scores']['test'][0]['main_score'])

        results_df = pd.DataFrame(scores, index=[0])

    if quantizer_name=="ProductQuantizer":
        results_df.to_json(os.path.join(output_folder_path, f'{model_name}_{matryoshka_trunc_dim}_dims', "PQ_NaiveBinarizer_IdChannel.json"))
    else:
        results_df.to_json(os.path.join(output_folder_path, f'{model_name}_{matryoshka_trunc_dim}_dims', "Truncated_IdChannel.json"))