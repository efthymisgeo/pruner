"""
Pruning a MLP by weights with one shot
"""
import argparse
import torch
import torch.nn as nn

from pruning.methods import weight_prune
from pruning.utils import learn_curves, train_eval_multi, load_config, test_multi,          iterative_pruning
from dataloaders.random_data import CreateRandomDataset
from models import MultiDNN, MultiDnnPruner

def main():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_config", required=False,
                        default="multilabel_dnn.json",
                        help="Model Architecture .json")
    parser.add_argument('-s', "--setup", default="multilabel_setup.json",
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
    test_acc, test_losses = test_multi(model, test_loader, criterion_list)

    for i, (val_loss, train_loss) in enumerate(zip(val_losses, train_losses)):
        learn_curves(train_loss, val_loss, "loss_fig_"+str(i)+".png")

    iterative_pruning(model, weight_pruner, criterion, data_loaders, prune_setup)

    ## Save and load the entire model
    ##torch.save(net.state_dict(), 'models/mlp_pruned.pkl')

if __name__ == "__main__":
    main()