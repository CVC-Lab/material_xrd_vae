import torch
import numpy as np
from torch import nn, optim
from model.SOSFlowNet import SOSFlowNet,BatchNormFlow,Reverse,FlowSequential
import matplotlib.pyplot as plt

def build_model(input_size, hidden_size, k, r, n_blocks, device=None, **kwargs):
    modules = []
    for i in range(n_blocks):
        modules += [
            SOSFlowNet(input_size, hidden_size, k, r),
            BatchNormFlow(input_size),
            Reverse(input_size)
        ]
    
    model = FlowSequential(*modules)
    if device is not None:
        model.to(device)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
   
    return model


class SOSFlow:
    def __init__(self, args):
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.k = args.num_polynomials # number of polynomials
        self.r = args.degree # degree if polynomials
        self.n_blocks = args.n_blocks
        self.num_epochs = args.epochs
        self.learning_rate = args.learning_rate
    

        self.device = torch.device("cuda:%d" % args.gpu if args.cuda else "cpu")
        self.network = build_model(self.input_size, self.hidden_size, self.k, self.r, self.n_blocks, self.device)
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=1e-6)


    def flow_loss(self, z, logdet, size_average=True, use_cuda=True):
        # If using Student-t as source distribution#
        #df = torch.tensor(5.0)
        #if use_cuda:
        #   log_prob = log_prob_st(z, torch.tensor([5.0]).cuda())
        #else:
            #log_prob = log_prob_st(z, torch.tensor([5.0]))
        #log_probs = log_prob.sum(-1, keepdim=True)
        ''' If using Uniform as source distribution
        log_probs = 0
        '''
        log_probs = (-0.5 * z.pow(2) - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        loss = -(log_probs + logdet).sum()
        # CHANGED TO UNIFORM SOURCE DISTRIBUTION
        #loss = -(logdet).sum()
        if size_average:
            loss /= z.size(0)
        return loss

        
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

        # accuracy = 0.
        num_batches = 0.
        
        # true_labels_list = []
        # predicted_labels_list = []
        


        # iterate over the dataset
        for (data, labels) in data_loader:
            data = data.to(self.device)
            labels= labels.long().to(self.device)

            optimizer.zero_grad()

            # flatten data
            # data = data.view(data.size(0), -1)
            
            # forward call
            zhat, log_jacob = self.network(data) 
            recon_loss_func = self.criterion(zhat, data)
            flow_loss_func = self.flow_loss(zhat, log_jacob, size_average=True)
            total_loss_func = recon_loss_func + flow_loss_func
            # accumulate values
            total_loss += total_loss_func.item()
            recon_loss += recon_loss_func.item()
            flow_loss += flow_loss_func.item()

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
        recon_loss /= num_batches
        flow_loss /= num_batches
        print('====> Epoch: {} Average loss per batch: {:.4f}; recon loss: {:.4f}; flow loss: {:.4f}\n'.format(epoch, total_loss, recon_loss, flow_loss))
        
        # # concat all true and predicted labels
        # true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
        # predicted_labels = torch.cat(predicted_labels_list, dim=0).cpu().numpy()

        # # compute metrics

        # accuracy = 100.0 * self.metrics.cluster_acc(predicted_labels, true_labels)
        # nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)

        return total_loss, recon_loss, flow_loss

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

        num_batches = 0
        accuracy = 0.
        
        true_labels_list = []
        predicted_labels_list = []

        criterion = nn.MSELoss(reduction='mean')

        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                labels= labels.long().to(self.device)
            
                # flatten data
                # data = data.view(data.size(0), -1)

                # forward call
                zhat, log_jacob = self.network(data) 
                recon_loss_func = self.criterion(zhat, data)
                flow_loss_func = self.flow_loss(zhat, log_jacob, size_average=True)
                # accumulate values
                total_loss += recon_loss_func.item() + flow_loss_func.item()
                recon_loss += recon_loss_func.item()
                flow_loss += flow_loss_func.item()

                num_batches += 1. 
        


        # average per batch
        if return_loss:
            total_loss /= num_batches
            recon_loss /= num_batches
            flow_loss /= num_batches
        
        print('====> Test Epoch: {} Average loss per batch: {:.4f}\n'.format(epoch, total_loss))

        if return_loss:
            return total_loss, recon_loss, flow_loss



    def train(self, train_loader, val_loader):
        """Train the model
        Args:
            train_loader: (DataLoader) corresponding loader containing the training data
            val_loader: (DataLoader) corresponding loader containing the validation data
        Returns:
            output: (dict) contains the history of train/val loss
        """
        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        train_history_err, val_history_err = [], []

        for epoch in range(1, self.num_epochs + 1):
            train_loss,_,_ = self.train_epoch(epoch, optimizer, train_loader)
            val_loss,_,_ = self.test(epoch, val_loader, True)
            train_history_err.append(train_loss)
            val_history_err.append(val_loss)


        return {'train_history_err' : train_history_err, 'val_history_err': val_history_err}