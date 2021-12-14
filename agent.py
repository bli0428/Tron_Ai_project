import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import numpy as np
import os
from hyperparameters import MODEL_PARAMETERS

log = logging.getLogger(__name__)

class Net(nn.Module):
    def __init__(self, state_size_row=17, state_size_col=17, action_size=4):
        """
            We define the network structure in this function
            You should have all the related parameters passed in with params, or you fix them for this specific game
        """
        super(Net, self).__init__()

        if torch.cuda.is_available():
            log.info('GPU found! Using CUDA for network...')
            self.device = 'cuda'
        else:
            log.info('GPU not found, using cpu instead...')
            self.device = 'cpu'
        self.params = MODEL_PARAMETERS
        self.num_epochs = MODEL_PARAMETERS["num_epochs"]
        self.batch_size = MODEL_PARAMETERS["batch_size"]
        self.network_size = MODEL_PARAMETERS["network_size"]
        self.dropout = MODEL_PARAMETERS["dropout"]

        self.state_size_row = state_size_row
        self.state_size_col = state_size_col
        self.state_size = state_size_row*state_size_col
        self.action_size = action_size

        self.conv1 = nn.Conv2d(1, self.network_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.network_size, self.network_size, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.network_size, self.network_size, 3, stride=1)
        self.conv4 = nn.Conv2d(self.network_size, self.network_size, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.network_size)
        self.bn2 = nn.BatchNorm2d(self.network_size)
        self.bn3 = nn.BatchNorm2d(self.network_size)
        self.bn4 = nn.BatchNorm2d(self.network_size)

        self.fc1 = nn.Linear(self.network_size*(self.state_size_col-4)*(self.state_size_row-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

        self.criterion = nn.MSELoss()

        self.params_dic = [{'params': self.parameters(), 'lr': self.params['learning_rate'], 'eps':1.5e-4}]

        self.optimizer = optim.Adam(self.params_dic)

        self.use_cuda = False

        if self.device == 'cuda':
            self.use_cuda = True

        self.to(self.device)


    def forward(self, s):
        """
        :param s: The state to evaluate
        :param a: The action to evaluate
        :return: The action value corresponding to that state and that action
        """
        # s: batch_size x state_size_col x state_size_row
        s = s.view(-1, 1, self.state_size_col, self.state_size_row)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.network_size*(self.state_size_col-4)*(self.state_size_row-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def train_model(self, examples):
        for epoch in range(self.num_epochs):
            # log.info(f'EPOCH ::: {epoch + 1}')
            self.train()

            sum_pi = 0
            sum_v = 0
            count = 0

            num_batches = int(len(examples) / self.batch_size)
            num_batches_count = range(num_batches)

            for batch in num_batches_count:
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.use_cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self(boards)
                pi_loss = self.loss_pi(target_pis, out_pi)
                v_loss = self.loss_v(target_vs, out_v)
                total_loss = pi_loss + v_loss

                # record loss
                sum_pi += pi_loss.item() * self.batch_size
                sum_v += v_loss.item() * self.batch_size
                count += self.batch_size

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
            # Since my output file was getting too big, I'm only allowing 1 logs of loss per EPOCH
            pi_losses = float(sum_pi) / count
            v_losses = float(sum_v) / count

            log.info(f'Total Loss: {total_loss:.2e}, Pi Loss: {pi_losses:.2e}, V Loss: {v_losses:.2e}')

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64))
        if self.use_cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.state_size_row, self.state_size_col)
        self.eval()
        with torch.no_grad():
            pi, v = self(board)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    # Try to search on google how to save and load a model
    def save(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.state_dict(),
        }, filepath)

    def load(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.use_cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])
