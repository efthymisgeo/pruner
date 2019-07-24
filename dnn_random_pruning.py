"""
Pruning a MLP by weights with one shot
"""
import argparse
import torch
import torch.nn as nn

from pruning.methods import weight_prune
from pruning.utils import learn_curves, train_eval, load_config, test, iterative_pruning
from dataloaders.random_data import CreateRandomDataset
from models import DNN, WeightPruner

def main():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_config", required=False,
                        default="deep_dnn.json",
                        help="Model Architecture .json")
    parser.add_argument('-s', "--setup", default="agressive_setup.json",
                        help="Experimental Setup .json")
    args = parser.parse_args()

    # config paths
    model_config = 'configs/model_architecture/' + args.model_config
    setup_path = "configs/experimental_setup/" + args.setup

    print(model_config)

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
    if datatype == "multilabel":
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
    model = DNN(config=model_config,
                in_features=feat_size,
                n_classes=n_classes)

    if torch.cuda.is_available():
        print('CUDA enabled.')
        model.cuda()
    print("--- DNN network initialized ---")
    print(model)

    # Criterion/Optimizer/Pruner
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weight_pruner = WeightPruner(model)

    # Train the model from scratch
    print("--- Training DNN ---")
    train_losses, val_losses = \
        train_eval(model, criterion, optimizer, epochs, train_loader, val_loader)
    test_acc, test_loss = test(model, test_loader, criterion)

    learn_curves(train_losses, val_losses, "loss_fig.png")

    iterative_pruning(model, weight_pruner, criterion, data_loaders, prune_setup)

    ###############################################################################
    ##### UNCOMMENT SINGLE # IF EVERYTHING IS RUINED
    ###############################################################################
    ## prune the weights
    ##masks = weight_prune(model, pruning_perc)
    ##model.set_masks(masks)
    ##weight_pruner = WeightPruner(model)
    #weight_pruner.pruning(pruning_perc)

    #print("--- {}% parameters pruned ---".format(pruning_perc))
    #test_acc, test_loss = test(model, test_loader, criterion)

    #prune_rate(model)

    ## Retraining
    ##criterion = nn.CrossEntropyLoss()
    ##optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'], 
    ##                                weight_decay=param['weight_decay'])

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    ##train(model, criterion, optimizer, param, loader_train)

    #print("--- Re-Training DNN ---")
    #train_losses, val_losses = \
    #    train_eval(model, criterion, optimizer, 300, train_loader, val_loader)
    ##test_acc, test_loss = test(model, test_loader, criterion)

    #learn_curves(train_losses, val_losses)


    ## Check accuracy and nonzeros weights in each layer
    #print("--- After retraining ---")
    #test_acc, test_loss = test(model, test_loader, criterion)
    #prune_rate(model)


    ## Save and load the entire model
    ##torch.save(net.state_dict(), 'models/mlp_pruned.pkl')

if __name__ == "__main__":
    main()