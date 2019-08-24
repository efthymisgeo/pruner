"""
Pruning a MLP by weights with one shot

Script for multilabel_setup.json & hierlabel_setup.json

Multilabel: 
            adds multiple uncorrelated labels in the feature vector
            and tries to predict each one in a different position of the network. In general it seems not to work well because the "tasks" are not similar and the labels are not corellated

Hirarchical Multilabel:
            we introduce this experiment to see that indeed deep nets build upon hierarchical entities. we expect it to work
            SubExp1 -> Inverse Hierarchy: addition/label1/label2
            SubExp2 -> Hierarchy: label_1/label_2/addition
"""
import argparse
import torch
import torch.nn as nn

from pruning.methods import weight_prune
from pruning.utils import learn_curves, train_eval_multi, load_config, test_multi, iterative_multi_pruning
from dataloaders.random_data import CreateRandomDataset
from models import MultiDNN, MultiDnnPruner

def dnn_prune(args):
    # config paths
    model_config = 'configs/model_architecture/' + args.model_config
    setup_path = "configs/experimental_setup/" + args.setup
    model_path = 'models/' + args.model_name
    pruned_model_path = 'models/pruned_' + args.model_name 

    # Hyper Parameters
    setup = load_config(setup_path)
    train_setup = setup["Train"]
    prune_setup = setup["Prune"]

    batch_size = train_setup["batch_size"]
    epochs = train_setup["training_epochs"]
    lr = train_setup["learning_rate"]
    datatype = train_setup["datatype"]
    feat_size = train_setup["feature_size"]
    n_samples = train_setup["n_samples"]
    n_classes = train_setup["n_classes"]
    val_ratio = train_setup["val_ratio"]
    test_ratio = train_setup["test_ratio"]
    
    labels = 1
    if datatype == "multilabel" or "hierlabel":
        labels = train_setup["labels_per_sample"]

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Dataloaders
    train_loader, val_loader, test_loader = \
        CreateRandomDataset(datatype,
                            feat_size,
                            n_samples,
                            n_classes,
                            val_ratio,
                            test_ratio,
                            batch_size,
                            labels).get_dataloaders()

    data_loaders = {"train": train_loader,
                    "val": val_loader, 
                    "test": test_loader}

    # Init model
    model = MultiDNN(config=model_config,
                     in_features=feat_size,
                     n_classes=n_classes,
                     n_labels=labels)

    model.load_state_dict(torch.load(model_path))                     

    if torch.cuda.is_available():
        print('CUDA enabled.')
        model.cuda()
    print("--- DNN network initialized ---")
    print(model)

    # Criterion/Optimizer/Pruner
    criterion = nn.CrossEntropyLoss()
    criterion_list = []
    for _ in range(labels):
        criterion_list.append(criterion)

    #get trainable parameters 
    #comment: model.parameters should also work!!!
    parameters = [params for params in model.parameters() if
                  params.requires_grad==True]
    optimizer = torch.optim.Adam(parameters,
                                 lr=lr)

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weight_pruner = MultiDnnPruner(model)

    print("--- Testing DNN ---")
    test_acc, test_losses = test_multi(model, test_loader, criterion_list)

    print("--- Start Pruning ---")

    model = iterative_multi_pruning(model, weight_pruner,
                                    criterion_list,
                                    data_loaders, prune_setup)

    torch.save(model.state_dict(), pruned_model_path)

def dnn_train(args):
    # config paths
    model_config = 'configs/model_architecture/' + args.model_config
    setup_path = "configs/experimental_setup/" + args.setup
    model_name = 'models/' + args.model_name

    # Hyper Parameters
    setup = load_config(setup_path)
    train_setup = setup["Train"]
    prune_setup = setup["Prune"]

    batch_size = train_setup["batch_size"]
    epochs = train_setup["training_epochs"]
    lr = train_setup["learning_rate"]
    datatype = train_setup["datatype"]
    feat_size = train_setup["feature_size"]
    n_samples = train_setup["n_samples"]
    n_classes = train_setup["n_classes"]
    val_ratio = train_setup["val_ratio"]
    test_ratio = train_setup["test_ratio"]
    
    labels = 1
    if datatype == "multilabel" or "hierlabel":
        labels = train_setup["labels_per_sample"]

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Dataloaders
    train_loader, val_loader, test_loader = \
        CreateRandomDataset(datatype,
                            feat_size,
                            n_samples,
                            n_classes,
                            val_ratio,
                            test_ratio,
                            batch_size,
                            labels).get_dataloaders()

    data_loaders = {"train": train_loader,
                    "val": val_loader, 
                    "test": test_loader}

    # Init model
    model = MultiDNN(config=model_config,
                     in_features=feat_size,
                     n_classes=n_classes,
                     n_labels=labels)

    if torch.cuda.is_available():
        print('CUDA enabled.')
        model.cuda()
    print("--- DNN network initialized ---")
    print(model)

    # Criterion/Optimizer/Pruner
    criterion = nn.CrossEntropyLoss()
    criterion_list = []
    for _ in range(labels):
        criterion_list.append(criterion)

    #get trainable parameters 
    #comment: model.parameters should also work!!!
    parameters = [params for params in model.parameters() if
                  params.requires_grad==True]
    optimizer = torch.optim.Adam(parameters,
                                 lr=lr)

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weight_pruner = MultiDnnPruner(model)

    # Train the model from scratch
    print("--- Training DNN ---")
    train_losses, val_losses = \
        train_eval_multi(model, criterion_list, optimizer,
                         epochs, train_loader, val_loader)
    
    print("--- Testing DNN ---")
    test_acc, test_losses = test_multi(model, test_loader, criterion_list)

    for i, (val_loss, train_loss) in enumerate(zip(val_losses, train_losses)):
        learn_curves(train_loss, val_loss, "loss_fig_"+str(i)+".png")

    #Save and load the entire model
    torch.save(model.state_dict(), model_name)  

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_config", required=False,
                        default="multilabel_dnn.json",
                        help="Model Architecture .json")
    parser.add_argument('-s', "--setup", default="multilabel_setup.json",
                        help="Experimental Setup .json")
    parser.add_argument('-c', "--model_name", required=True,
                        default='dnn_sum_hierlabel_exp1.pth')

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--train', action='store_true', help='Train model')
    action.add_argument('--prune', action='store_true', help='Prune model')

    args = parser.parse_args()

    if args.train:
        dnn_train(args)

    if args.prune:
        dnn_prune(args)        