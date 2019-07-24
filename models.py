import numpy as np
import torch
import torch.nn as nn
from pruning.layers import MaskedLinear, MaskedConv2d
from pruning.utils import load_config


class DNN(nn.Module):
    """Deep Neural Network architecture.

    Args:
        config (str): Config with the parameters. It can be a
            json file path or a dict.
        input_size (tuple): Input size
    """
    def __init__(self, config="config.json",
                 in_features=None, n_classes=3):
        super(DNN, self).__init__()

        self.config = load_config(config)
        self.in_features = in_features
        self.out_size = n_classes

        # Configuration parameters for DNN layers
        self.dnn_config = self.config["DNN"]
        self.layers = self.dnn_config["layers"]
        self.units = self.dnn_config["units"]
        self.activation = self.dnn_config["activation"]
        self.dropout = self.dnn_config["dropout"]
        self.drop_prob = self.dnn_config["drop_prob"]

        # self.classifier = self._create_fc_layers(input_size)
        self.dnn = self._create_layers()

    def _create_layers(self):
        """Construct the fully-connected layers.
        """
        modules_dnn = []
        inp_size = self.in_features

        for i_fc in range(self.layers):
            modules_dnn.append(MaskedLinear(int(inp_size),
                               self.units[i_fc]))
            modules_dnn.append(nn.ReLU())
            if self.dropout:
                modules_dnn.append(nn.Dropout(p=self.drop_prob[i_fc]))
            inp_size = self.units[i_fc]

        modules_dnn.append(MaskedLinear(inp_size, self.out_size))

        return nn.Sequential(*modules_dnn)

    def forward(self, x):
        """Feed-forward function
        """
        out = self.dnn(x)
        return out

class MultiDNN(nn.Module):
    def __init__(self, config="multilabel.json",
                 in_features=300,
                 n_classes=3, n_labels=4):
        super(MultiDNN, self).__init__()
       
        self.config = load_config(config)
        self.in_features = in_features
        self.out_size = n_classes
        self.n_modules = n_labels

        # Configuration parameters for DNN layers
        self.mods = self.config["modules"]
        dnn_list, clf_layers = self._load_modules()
        self.module_dict = nn.ModuleDict([
                ['dnn_list', dnn_list],
                ['clf_layers', clf_layers]
        ])

    def _load_modules(self):
        mod_name = "DNN_" # the .json prefix
        mod_list = []
        for i_mod in range(1, self.n_modules+1):
            mod_list.append(self.mods[mod_name + str(i_mod)])

        dnn_list = nn.ModuleList()
        clf_layers = nn.ModuleList()
        n_dnn = len(mod_list) - 1

        in_features = self.in_features
        #out_size = self.out_size
        
        for i, mod in enumerate(mod_list):
            layers = mod["layers"]
            units = mod["units"]
            #activation = mod["activation"]
            dropout = mod["dropout"]
            drop_prob = mod["drop_prob"]
            dnn, clf_layer = self.create_layers(in_features,
                                                self.out_size,
                                                layers, units,
                                                dropout, drop_prob)
            dnn_list.append(dnn)

            in_features = units[-1]
            
            clf_layers.append(clf_layer)

        return dnn_list, clf_layers
    
    @staticmethod
    def create_layers(in_features, out_size, layers,
                      units, dropout, drop_prob):
        """Construct the fully-connected layers.
           Output:
                modules_dnn: nn.Sequential dnn
                clf_layer: the auxiliary loss function layers
        """
        modules_dnn = []
        inp_size = in_features

        for i_fc in range(layers):
            modules_dnn.append(MaskedLinear(int(inp_size),
                               units[i_fc]))
            modules_dnn.append(nn.ReLU())
            if dropout:
                modules_dnn.append(nn.Dropout(p=drop_prob[i_fc]))
            inp_size = units[i_fc]

        clf_layer = MaskedLinear(inp_size, out_size)

        return nn.Sequential(*modules_dnn), clf_layer

    def forward(self, x):
        """Feed-forward function
        """
        out = []
        aux_clf_layers = self.n_modules - 1

        #fix me
        
        for i, dnn in enumerate(self.module_dict["dnn_list"]):
            y = dnn(x)
            if i == aux_clf_layers:
                out.append(y)
            else:
                out.append(self.module_dict["clf_layers"][i](y))

            x = y
        return out 



class Pruner():
    def __init__(self, model):
        self.model = model
        self.masks = []

    def pruning(self):
        # method to be overriden
        pass



class MultiDnnPruner(Pruner):
    def __init__(self, model):
        super(MultiDnnPruner, self).__init__(model)
        self.model = self.get_prunable(model)
        print(self.model)

    def pruning(self, pruning_perc):
        self._weight_prune(pruning_perc)
        self._set_masks()
        return(self.masks)   

    def _weight_prune(self, pruning_perc):
        '''
        Prune pruning_perc% weights globally (not layer-wise)
        arXiv: 1606.09274
        '''
        all_weights = []
        if not self.masks:
            # first time to prune
            for p in self.model.parameters():
              if len(p.data.size()) != 1:
                  all_weights += list(p.cpu().data.abs().numpy().flatten())
            threshold = np.percentile(np.array(all_weights), pruning_perc)
        else:
            # iterative pruning
            for p in self.model.parameters():
                if len(p.data.size()) != 1:
                    weights = p.cpu().data.abs().numpy().flatten()
                    #nonzero_weights = weights[np.nonzero(weights)]
                    all_weights += list(weights[np.nonzero(weights)])
            threshold = np.percentile(np.array(all_weights), pruning_perc)

        # generate mask
        self.masks = []
        for p in self.model.parameters():
            if len(p.data.size()) != 1:
                pruned_inds = p.data.abs() > threshold
                self.masks.append(pruned_inds.float())

        return()
    
    def _set_masks(self):
        """Applies mask is every linear layer
        """
        i_fc = 0
        for p in self.model.modules():
            if isinstance(p, MaskedLinear):
                p.set_mask(self.masks[i_fc])
                i_fc += 1
        return()
    
    @staticmethod
    def get_prunable(model):
        dnn_model = model.module_dict["dnn_list"]
        seq_model = []
        for p in dnn_model:
            if isinstance(p, nn.Sequential):
                for q in p.children():
                    seq_model.append(q)
        
        last_clf = model.module_dict["clf_layers"][-1]
        seq_model.append(last_clf)

        return(nn.Sequential(*seq_model))


    
    def print_mask(self):
        for i in self.masks:
            print(i)


class WeightPruner(Pruner):
    def __init__(self, model):
        super(WeightPruner, self).__init__(model)
        #self.model = model
        #self.masks = []

    def pruning(self, pruning_perc):
        self._weight_prune(pruning_perc)
        self._set_masks()
        return(self.masks)   

    def _weight_prune(self, pruning_perc):
        '''
        Prune pruning_perc% weights globally (not layer-wise)
        arXiv: 1606.09274
        '''
        all_weights = []
        if not self.masks:
            # first time to prune
            for p in self.model.parameters():
              if len(p.data.size()) != 1:
                  all_weights += list(p.cpu().data.abs().numpy().flatten())
            threshold = np.percentile(np.array(all_weights), pruning_perc)
        else:
            # iterative pruning
            for p in self.model.parameters():
                if len(p.data.size()) != 1:
                    weights = p.cpu().data.abs().numpy().flatten()
                    #nonzero_weights = weights[np.nonzero(weights)]
                    all_weights += list(weights[np.nonzero(weights)])
            threshold = np.percentile(np.array(all_weights), pruning_perc)

        # generate mask
        self.masks = []
        for p in self.model.parameters():
            if len(p.data.size()) != 1:
                pruned_inds = p.data.abs() > threshold
                self.masks.append(pruned_inds.float())

        return()
    
    def _set_masks(self):
        """Applies mask is every linear layer
        """
        i_fc = 0
        for p in self.model.modules():
            if isinstance(p, MaskedLinear):
                p.set_mask(self.masks[i_fc])
                i_fc += 1
        return()    
    
    def _check_model_architect(self):
        """Checks if we have a DNN or a multiDNN
        """

    
    def print_mask(self):
        for i in self.masks:
            print(i)




class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = MaskedLinear(28*28, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MaskedLinear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = MaskedLinear(200, 10)
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.linear1.set_mask(masks[0])
        self.linear2.set_mask(masks[1])
        self.linear3.set_mask(masks[2])
    

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = MaskedConv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = MaskedConv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = MaskedConv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(7*7*64, 10)
        
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.relu3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.conv1.set_mask(torch.from_numpy(masks[0]))
        self.conv2.set_mask(torch.from_numpy(masks[1]))
        self.conv3.set_mask(torch.from_numpy(masks[2]))
