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
import torch


class Binarizer(torch.nn.Module):
    """A Binarizer transforms tensor values into binary representations.

    Args:
        bits (integer): size in bits of integer representations.

    Raises:
        NotImplementedError: this classed is meant to be subclassed
    """
    
    def __init__(self, bits) -> None:
        super(Binarizer, self).__init__()
        self.bits = bits
                
    def encode(self, x:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def decode(self, x:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.encode(x)