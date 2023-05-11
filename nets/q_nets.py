import torch
from torch import nn
from nets.customlayers import (
    activation_func,
    )



class Simple_QL_FC(nn.Module):
    """
    A simple fully-connected decision maker.
    """
   
    def __init__(self, params):
        """
        Args:
            params (Dictionary):
                in_size:        Input vector dimensionality
                n_features:     List of numbers of features per each layer
                out_size:       Output vector dimensionality
        """

        super().__init__()
        in_size = params['in_size']
        n_features = params['n_features']
        out_size = params['out_size']
        n_features_tot = [in_size] + n_features + [out_size]

        n_hidden = len(n_features)

        layers = []

        for fdx in range(n_hidden):
            curr_in_size = n_features_tot[fdx]
            curr_out_size = n_features_tot[fdx+1]
            layers.append(nn.Linear(curr_in_size, curr_out_size))
            layers.append(activation_func('relu'))
        layers.append(nn.Linear(n_features_tot[-2], n_features_tot[-1]))
        #layers.append
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        q_values = self.net(x)
        return q_values
