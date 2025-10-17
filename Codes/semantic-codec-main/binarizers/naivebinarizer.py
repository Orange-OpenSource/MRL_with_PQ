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

from binarizers import Binarizer


class NaiveBinarizer(Binarizer):
    
    def __init__(self, bits) -> None:
        """The NaiveBinarizer transforms integer tensor values into binary representations, most significant bits on the left.

        Args:
            bits (integer): size in bits of integer representations

        """
        super(NaiveBinarizer, self).__init__(bits=bits) # self.bits = bits
        self.register_buffer(name='mask', tensor=2**torch.arange(self.bits-1, -1, -1, device='cpu', dtype=torch.int32), persistent=False) # dtype = torch.int32 | shape = [self.bits]
    
    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """Generate binary representation of integer tensor.

        Args:
            x (torch.Tensor) [dtype torch.int32|int64]: integer tensor, ex. torch.tensor(data=[1,2,9,10,17], dtype=torch.int64|int32)

        Returns:
            (torch.Tensor) [dtype torch.int32]: each line at [...,i,:self.bits] contains binary representation of integer i
        """
        bits = x.unsqueeze(-1).bitwise_and(self.mask).ne(0).to(torch.int32) # dtype = torch.int32 | shape = [...,x.size,bits]
        return bits
    
    def decode(self, x:torch.Tensor) -> torch.Tensor:
        """From binary representation, returns the integer values.

        Args:
            x (torch.Tensor) [dtype torch.int32|int64]: binary representation, each line at [...,i,:bits] contains binary representation of integer i

        Returns:
            (torch.Tensor): type depends on x data type, can be int or float.
                            last dimension of input is removed and integers are represented as tensor([1,2,9,10,17])
        """
        integers = torch.sum(self.mask * x, -1)
        return integers
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        bits = self.encode(x)
        return bits