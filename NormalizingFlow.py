import torch
import numpy as np
from torch import nn, optim
from model.NormalizingFlowNet import VAENF, PlanarTransformation, HouseholderSylvesterTransformation, TriagSylvesterTransformation
from loss_function import VAENFLoss

class NormalizingFlow:
    def __init__(self, args):
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.n_blocks = args.n_blocks
        self.num_epochs = args.epochs
        self.learning_rate = args.learning_rate
    
        self.device = torch.device("cuda:%d" % args.gpuID if args.cuda else "cpu")
        self.network = VAENF(self.input_size, self.hidden_size, PlanarTransformation, self.hidden_size, self.n_blocks)
        if self.network is not None:
            self.network.to(self.device)
        self.criterion = VAENFLoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=1e-6)

        
    def train_epoch(self, epoch, optimizer, data_loader):
        """Train the model for one epoch
        Args:
            optimizer: (Optim) optimizer to use in backpropagation
            data_loader: (DataLoader) corresponding loader containing the training data
        Returns:
            average of all loss values, accuracy, nmi
        """
        self.network.train()
        total_loss = 0.
        recon_loss = 0.
        flow_loss = 0.
        energy_loss = 0.

        # accuracy = 0.
        num_batches = 0.
        
        # true_labels_list = []
        # predicted_labels_list = []
        


        # iterate over the dataset
        for (data, e) in data_loader:
            data = data.to(self.device)
            e = e.view(-1,1).to(self.device)

            optimizer.zero_grad()

            # flatten data
            # data = data.view(data.size(0), -1)
            
            # forward call
            zhat, mu, logvar, e_pred = self.network(data, energy=True) 
            log_jacob = self.network.flow.get_sum_log_det()
          
            total_loss_func = self.criterion.VAENF_loss(zhat, data, mu, logvar, log_jacob)
            energy_loss_func = self.criterion.energy_loss(e_pred, e)
            # accumulate values
            total_loss += total_loss_func.item()
            energy_loss += energy_loss_func.item()

            # perform backpropagation
            total_loss_func.backward()
            optimizer.step()  

            # # save predicted and true labels
            # predicted = unlab_loss_dic['predicted_labels']

            # true_labels_list.append(labels)
            # predicted_labels_list.append(predicted)   
        
            num_batches += 1. 
            if num_batches % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, num_batches * len(data), len(data_loader.dataset),
                    100. * num_batches / len(data_loader),  total_loss_func.item()))



        # average per batch
        total_loss /= num_batches
        energy_loss /= len(data_loader.dataset)
        print('====> Epoch: {} Average loss per batch: {:.4f};\n'.format(epoch, total_loss))
        
        # # concat all true and predicted labels
        # true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
        # predicted_labels = torch.cat(predicted_labels_list, dim=0).cpu().numpy()

        # # compute metrics

        # accuracy = 100.0 * self.metrics.cluster_acc(predicted_labels, true_labels)
        # nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)

        return total_loss, recon_loss, flow_loss, energy_loss

    def test(self, epoch, data_loader, return_loss=False):
        """Test the model with new data
        Args:
            data_loader: (DataLoader) corresponding loader containing the test/validation data
            return_loss: (boolean) whether to return the average loss values
            
        Return:
            accuracy and nmi for the given test data
        """
        self.network.eval()
        total_loss = 0.
        recon_loss = 0.
        flow_loss = 0.
        energy_loss = 0.

        num_batches = 0
        accuracy = 0.
        

        with torch.no_grad():
            for data, e in data_loader:
                data = data.to(self.device)
                e = e.view(-1,1).to(self.device)
            
                # flatten data
                # data = data.view(data.size(0), -1)

                # forward call
                
                zhat, mu, logvar, e_pred = self.network(data, energy=True) 
                log_jacob = self.network.flow.get_sum_log_det()
                total_loss_func = self.criterion.VAENF_loss(zhat, data, mu, logvar, log_jacob)
                energy_loss_func = self.criterion.energy_loss(e_pred, e)
                # accumulate values
                total_loss += total_loss_func.item()
                energy_loss += energy_loss_func.item()
                num_batches += 1. 
        


        # average per batch
        if return_loss:
            total_loss /= num_batches
            energy_loss /= len(data_loader.dataset)
        
        print('====> Test Epoch: {} Average loss per batch: {:.4f}\n'.format(epoch, total_loss))

        if return_loss:
            return total_loss, recon_loss, flow_loss, energy_loss



    def train(self, train_loader, val_loader):
        """Train the model
        Args:
            train_loader: (DataLoader) corresponding loader containing the training data
            val_loader: (DataLoader) corresponding loader containing the validation data
        Returns:
            output: (dict) contains the history of train/val loss
        """
        train_history_err, val_history_err = [], []

        for epoch in range(1, self.num_epochs + 1):
            train_loss,_,_,energy_loss = self.train_epoch(epoch, self.optimizer, train_loader)
            val_loss,_,_,val_energy_loss = self.test(epoch, val_loader, True)
            train_history_err.append(energy_loss)
            val_history_err.append(val_energy_loss)


        return {'train_history_err' : train_history_err, 'val_history_err': val_history_err}