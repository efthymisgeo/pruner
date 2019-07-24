import six
import json
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler
import matplotlib.pyplot as plt


def learn_curves(train,val,name):
    plt.figure()
    plt.title("Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(train, 'r', label='training curve')
    plt.plot(val, 'g', label='validation curve')
    plt.legend()
    plt.savefig(name)

def load_config(config):
    """Load configuration.
    """
    if isinstance(config, six.string_types):
        with open(config, "r") as f:
            return json.load(f)
    elif isinstance(config, dict):
        return config
    else:
        raise NotImplementedError("Config must be a json file or a dict")


def to_var(x, requires_grad=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if isinstance(x, list):
        for k in x:
            if torch.cuda.is_available():
                k = k.cuda()
            k = torch.tensor(k, requires_grad=requires_grad)
    else:
        if torch.cuda.is_available():
                x = x.cuda()
        x = torch.tensor(x, requires_grad=requires_grad)

    return x


def train_eval(model, loss_fn, optimizer, n_epochs, loader_train, loader_val):
    epoch_loss_train = []
    epoch_loss_val = []
    for epoch in range(n_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, n_epochs))
        
        train_loss = train(model, loss_fn, optimizer, loader_train)
        acc, val_loss = test(model, loader_val, loss_fn)

        epoch_loss_train.append(train_loss)
        epoch_loss_val.append(val_loss)

    return epoch_loss_train, epoch_loss_val

def train_eval_multi(model, loss_fn, optimizer, n_epochs, loader_train, loader_val):
    epoch_loss_train = [[] for _ in loss_fn]
    epoch_loss_val = [[] for _ in loss_fn]

    for epoch in range(n_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, n_epochs))
        
        train_loss_list = train_multi(model, loss_fn, optimizer, loader_train)
        acc_list, val_loss_list = test_multi(model, loader_val, loss_fn)

        epoch_loss_train = append_list(train_loss_list, epoch_loss_train)
        epoch_loss_val = append_list(val_loss_list, epoch_loss_val)

    return epoch_loss_train, epoch_loss_val    

def append_list(list_a, list_b):
    """ Append list of -n- elements in list
        of -n- lists
    """
    for i, a in enumerate(list_a):
        list_b[i].append(a)

    return list_b

def to_long(mylist):
    for i, a in enumerate(mylist):
        mylist[i] = a.long()
    
    return(mylist)
        
    

def train_multi(model, loss_list, optimizer, loader_train):
    model.train()
    # in case of an error uncomment
    # model.module_dict["dnn_list"].train()
    # model.module_dict["clf_layers"].train()
    
    running_loss = [0 for _ in loss_list]

    for t, (x, y) in enumerate(loader_train):
        y = list(map(to_long, y))
        x_var, y_var = to_var(x.float()), to_var(y)

        scores_list = model(x_var)

        # r u sure optim.zero_grad can go here??
        optimizer.zero_grad()
        
        for i, (loss_fn, scores, y_gold) in enumerate(zip(loss_list,
                                                          scores_list,
                                                          y_var)):
            loss = loss_fn(scores, y_gold)
            running_loss[i] += loss.item()

            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.item()))

            
            loss.backward(retain_graph=True)
        
        optimizer.step()
    
    return [run_loss/len(loader_train.dataset) for run_loss in running_loss]

def test_multi(model, loader, loss_list):

    model.eval() #same as train_multi for this
    
    n_labels = len(loss_list)

    test_loss = [0 for _ in range(n_labels)]
    num_correct = [0 for _ in range(n_labels)]
    num_samples = len(loader.dataset)

    test_loss = [0 for _ in range(n_labels)]
    num_correct = [0 for _ in range(n_labels)]
    acc = [0 for _ in range(n_labels)]

    for t, (x, y) in enumerate(loader):
        y = list(map(to_long, y))
        x_var, y_var = to_var(x.float()), to_var(y)

        scores_list = model(x_var)
        for i, (loss_fn, scores, y_gold) in enumerate(zip(loss_list,
                                                        scores_list,
                                                        y_var)):
            
            loss = loss_fn(scores, y_gold)
            test_loss[i] += loss.item()

            _, preds = scores.data.cpu().max(1)
            num_correct[i] += (preds == y_gold).sum()

    for i, corrects in enumerate(num_correct):
        acc[i] = float(corrects) / num_samples

        print('Test accuracy for label {} out of {}: {:.2f}% ({}/{})'.format(
            i+1,
            n_labels,
            100.*acc[i],
            corrects,
            num_samples,
            ))
    
    return acc, [t/len(loader.dataset) for t in test_loss]    

    

def train(model, loss_fn, optimizer, loader_train):

    model.train()
    running_loss = 0
    for t, (x, y) in enumerate(loader_train):
        x_var, y_var = to_var(x.float()), to_var(y.long())
        scores = model(x_var)
        loss = loss_fn(scores, y_var)
        running_loss += loss.item()

        if (t + 1) % 100 == 0:
            print('t = %d, loss = %.8f' % (t + 1, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return running_loss/len(loader_train.dataset)
         

def test(model, loader, loss_fn):

    model.eval()
    test_loss = 0
    num_correct, num_samples = 0, len(loader.dataset)
    for t, (x, y) in enumerate(loader):
        x_var = to_var(x.float())
        y_var = to_var(y.long())

        scores = model(x_var)
        loss = loss_fn(scores, y_var)
        
        test_loss += loss.item()

        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
        ))
    
    return acc, test_loss/len(loader.dataset)

def iterative_pruning(model, pruner, criterion, data_loaders, setup):
    '''Takes model pruner and list of setup values s.a
        optimizers and learning rates that will be applied 
        to each pruning round
    '''
    
    pruning_perc = setup["pruning_perc"]
    #optimizers = setup["optimizers"]
    optimizer = setup["optimizers"]
    l_rates = setup["learning_rates"]
    retrain_epochs = setup["retrain_epochs"]
    pruning_rounds = setup["pruning_rounds"]

    for r, (prune_ratio, lr, epochs)  in enumerate(zip(pruning_perc,
                                                       l_rates,
                                                       retrain_epochs)):
        print("--- Pruning Round: {}/{} ---".format(r,pruning_rounds))
        pruned_model = one_shot_pruning(model, pruner, criterion, prune_ratio,
                         optimizer, lr, epochs, data_loaders)
        model = pruned_model
        pruner.print_mask()
    return()

def one_shot_pruning(model, pruner, criterion, pruning_perc,
                     optim_name, lr, epochs, data_loaders):
    
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    test_loader = data_loaders["test"]

    # prune weights
    masks = pruner.pruning(pruning_perc)
    #model.set_masks(masks)
    print("--- {}% parameters pruned ---".format(pruning_perc))
    _, _ = test(model, test_loader, criterion)
    prune_rate(model)

    # Retraining
    if optim_name=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError()
    
    print("--- Re-Training DNN ---")
    train_losses, val_losses = \
        train_eval(model, criterion, optimizer, epochs, train_loader, val_loader)
    learn_curves(train_losses, val_losses, "pruning_loss.png")
    
    # Check accuracy and nonzeros weights in each layer
    print("--- After retraining ---")
    _, _ = test(model, test_loader, criterion)
    prune_rate(model)
    return(model)
    

def prune_rate(model, verbose=True):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 \
                            else 'Linear',
                        100.*zero_param_this_layer/param_this_layer,
                        ))
    pruning_perc = 100.*nb_zero_param/total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc


def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
         if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix
    