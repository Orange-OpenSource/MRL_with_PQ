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
from abc import abstractmethod
from typing import List, Union
import torch

class Channel(torch.nn.Module):
    
    def __init__(self) -> None:

        super(Channel, self).__init__()

    @abstractmethod        
    def apply(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """apply channel to input

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): input

        Raises:
            NotImplementedError: this method must be implemented

        Returns:
            torch.Tensor: channel output
        """
        raise NotImplementedError
        
    def forward(self, x):
        """forward call of the model

        Args:
            x (torch.Tensor): channel input

        Returns:
            torch.Tensor: channel output
        """
        self.apply(x)


class IdentityChannel(Channel):
    
    def __init__(self) -> None:

        super(IdentityChannel, self).__init__()

    def to(self, device):
        super(IdentityChannel, self).to(device)
        return self

                
    def apply(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        return x