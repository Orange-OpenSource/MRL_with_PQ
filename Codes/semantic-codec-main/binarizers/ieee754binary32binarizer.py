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


class IEEE754Binary32Binarizer(Binarizer):

    def __init__(self) -> None:
        super(IEEE754Binary32Binarizer, self).__init__(bits=32) # self.bits = 32
                
        self.register_buffer(name='exponent_bitlength', tensor=torch.tensor(data=8, device='cpu', dtype=torch.int32), persistent=False) # dtype = torch.int32
        self.register_buffer(name='mantissa_bitlength', tensor=torch.tensor(data=23, device='cpu', dtype=torch.int32), persistent=False) # dtype = torch.int32
        self.register_buffer(name='exponent_bias', tensor=torch.tensor(data=127, device='cpu', dtype=torch.int32), persistent=False) # dtype = torch.int32
        
        self.register_buffer(name='exponent_mask', tensor=2**torch.arange(self.exponent_bitlength-1, -1, -1, dtype=torch.int32), persistent=False) # dtype = torch.int32 | shape = [exponent_bitlength=8]
        self.register_buffer(name='decimal_mask', tensor=2**(1+torch.arange(self.mantissa_bitlength, dtype=torch.int32)), persistent=False) # dtype = torch.int32 | shape = [mantissa_bitlength=23]
        self.register_buffer(name='decimal_reverse_mask', tensor=2.**-(1+torch.arange(self.mantissa_bitlength, dtype=torch.int32)), persistent=False) # dtype = torch.float32 | shape = [mantissa_bitlength=23]

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """Generate IEEE.754 binary representation of a float tensor.

        Args:
            x (torch.Tensor) [dtype torch.float32|float64]: float tensor, ex. torch.tensor(data=[1.,2.,9.,10.,17.], dtype=torch.float64|float32)

        Returns:
            ieee754_binary32_bits (torch.Tensor) [dtype torch.int32]: each line at [...,i,:self.bits=32] contains binary representation of integer i
        """
        sign_bits = torch.where(x < 0, 1, 0).to(torch.int32)
        unsigned_x = x * (-1)**sign_bits
        exponent_x = torch.log2(unsigned_x).floor().to(torch.int32)
        exponent_x = torch.where(exponent_x < -self.exponent_bias, -self.exponent_bias, exponent_x) # to deal with x=0.
        biased_exponent_x = exponent_x + self.exponent_bias

        exponent_bits = biased_exponent_x.unsqueeze(-1).bitwise_and(self.exponent_mask).ne(0).to(torch.int32)
                
        exponent_x = torch.where(exponent_x == -self.exponent_bias, -self.exponent_bias+1, exponent_x) # to deal with exponent -127 on MPS device
        fractional_x = torch.divide(unsigned_x, 2.**exponent_x)
        decimal_x = fractional_x % 1
        
        mantissa_bits = (torch.multiply(decimal_x.unsqueeze(-1), self.decimal_mask) % 2).to(torch.int32)
        
        ieee754_binary32_bits = torch.concatenate((sign_bits.unsqueeze(-1), exponent_bits, mantissa_bits), dim=-1)
        return ieee754_binary32_bits
    
    def decode(self, ieee754_binary32_bits:torch.Tensor) -> torch.Tensor:
        """From IEEE 754 binary representation, returns the float values.

        Args:
            x (torch.Tensor) [dtype torch.int32|int64]: binary representation, each line at [...,i,:bits=32] contains IEEE 754 binary representation of float i

        Returns:
            (torch.Tensor): last dimension of input is removed and floats are represented as tensor([1.,-2.,9.,0.10,-1.7])
        """
        bit_signs = ieee754_binary32_bits[...,0]
        exponent_bits = ieee754_binary32_bits[...,1:9]
        mantissa_bits = ieee754_binary32_bits[...,9:]

        signs = (-1)**bit_signs
        
        exponents = torch.sum(self.exponent_mask * exponent_bits, -1) - self.exponent_bias

        decimals = torch.sum(self.decimal_reverse_mask * mantissa_bits, -1)
        fractionals = torch.where(exponents == -127, decimals, 1. + decimals)
        
        exponents = torch.where(exponents == -127, -126, exponents) # subnormal values
        exponents = torch.where(exponents == 128, 127, exponents) # infinity/NaN values clipped to max exponent value (127)
        
        floats = signs * (2.**exponents) * fractionals
        return floats
            
    def forward(self, x):
        bits = self.encode(x)
        return bits
