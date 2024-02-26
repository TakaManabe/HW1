__author__ = "Takahiro Manabe"

import torch.nn as nn # レイヤーを構成するためのライブラリ
from torchsummary import summary

################################################################################################################
################################################################################################################
################################################################################################################
class Model_1(nn.Module):
    # 使用するオブジェクトを定義
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following stmt (statement) with appropriate input dimensions once model's implementation is done.
        self.hidden = nn.Linear(input_dim, hidden_size) # Hidden layers -> 100
        self.output = nn.Linear(hidden_size, 10) # MNIST label num = 10!!

    # 順伝搬
    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        
        x = torch.sigmoid(self.hidden(x))
        features = self.output(x)
        return features

################################################################################################################
################################################################################################################
################################################################################################################
class Model_2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # receptive field 5x5, stride 1?
        self.Conv2d_1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=5, stride=1, padding=2)
        self.Conv2d_2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=5, stride=1, padding=2)
        
        # 2x2 kernel, 1 stride (can be one pooling definition though)
        self.Pool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Pool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Here input_dim is not defined as an argument: calculate
        self.hidden = nn.Linear(in_features= 1960, out_features=hidden_size) # Hidden layers
        self.output = nn.Linear(hidden_size, 10) # MNIST label num = 10!!

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------

        x = nn.ReLU()(self.Conv2d_1(x)) # Conv + ReLU
        x = self.Pool2d_1(x) # Pool
        x = nn.ReLU()(self.Conv2d_2(x)) # Conv + ReLU
        x = self.Pool2d_2(x) # Pool

        x = x.view(x.size(0), -1) # Flatten

        # From model 1
        x = torch.sigmoid(self.hidden(x))
        features = self.output(x)
        # Uncomment the following return stmt once method implementation is done.
        return features

################################################################################################################
################################################################################################################
################################################################################################################
class Model_3(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # receptive field 5x5, stride 1?
        self.Conv2d_1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=5, stride=1, padding=2)
        self.Conv2d_2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=5, stride=1, padding=2)
        
        # 2x2 kernel, 1 stride
        self.Pool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Pool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Here input_dim is not defined as an argument: calculate
        self.hidden = nn.Linear(in_features= 1960, out_features=hidden_size) # Hidden layers
        self.output = nn.Linear(hidden_size, 10) # MNIST label num = 10!!

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  features
        # Delete line return NotImplementedError() once method is implemented.
        x = nn.ReLU()(self.Conv2d_1(x)) # Conv + ReLU
        x = self.Pool2d_1(x)
        x = nn.ReLU()(self.Conv2d_2(x)) # Conv + ReLU
        x = self.Pool2d_2(x)

        x = x.view(x.size(0), -1)

        # From model 1
        x = nn.ReLU(self.hidden(x))
        # x6 = torch.sigmoid(self.hidden(x5))
        features = self.output(x)

        # Uncomment the following return stmt once method implementation is done.
        return features


################################################################################################################
################################################################################################################
################################################################################################################
class Model_4(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------

        # receptive field 5x5, stride 1?
        self.Conv2d_1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=5, stride=1, padding=2)
        self.Conv2d_2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=5, stride=1, padding=2)
        
        # 2x2 kernel, 1 stride
        self.Pool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Pool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Here input_dim is not defined as an argument: calculate
        self.hidden = nn.Linear(in_features= 1960, out_features=hidden_size) # Hidden layers
        self.hidden_2 = nn.Linear(hidden_size, 100) # Hidden layers

        self.output = nn.Linear(100, 10) # MNIST label num = 10!!

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        x = nn.ReLU()(self.Conv2d_1(x)) # Conv + ReLU
        x = self.Pool2d_1(x)
        x = nn.ReLU()(self.Conv2d_2(x)) # Conv + ReLU
        x = self.Pool2d_2(x)

        x = x.view(x.size(0), -1)

        # FCs
        # L2 regularization on the parameters of the two FC layers
        x = nn.ReLU(self.hidden(x))
        x = nn.ReLU(self.hidden_2(x))
        features = self.output(x)

        return features


################################################################################################################
################################################################################################################
################################################################################################################
class Model_5(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # receptive field 5x5, stride 1?
        self.Conv2d_1 = nn.Conv2d(out_channels=40, kernel_size=5, stride=1, padding='same')
        self.Conv2d_2 = nn.Conv2d(out_channels=40, kernel_size=5, stride=1, padding='same')
        
        # 2x2 kernel, 1 stride
        self.Pool2d_1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.Pool2d_2 = nn.MaxPool2d(kernel_size=2, stride=1)


        # Here input_dim is not defined as an argument: calculate
        self.hidden = nn.Linear(in_features= 40 * 26 * 26, out_features=hidden_size) # Hidden layers
        self.hidden_2 = nn.Linear(hidden_size, 1000) # Hidden layers
        self.output = nn.Linear(1000, 10) # MNIST label num = 10!!
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features   #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        x2 = self.ReLU(self.Conv2d_1(x)) # Conv + ReLU
        x3 = self.Pool2d_1(x)
        x4 = self.ReLU(self.Conv2d_2(x)) # Conv + ReLU
        x5 = self.Pool2d_2(x)

        # FCs
        x6 = self.ReLU(self.hidden(x5))
        x7 = self.ReLU(self.hidden_2(x6))
        features = nn.Dropout(p=0.5)(self.output(x7))

        return features


################################################################################################################
################################################################################################################
################################################################################################################
class Net(nn.Module):
    def __init__(self, mode, args):
        super().__init__()
        self.mode = mode
        self.hidden_size= args.hidden_size
        # model 1: base line
        if mode == 1:
            in_dim = 28*28 # input image size is 28x28
            self.model = Model_1(in_dim, self.hidden_size)

        # model 2: use two convolutional layer
        if mode == 2:
            self.model = Model_2(self.hidden_size)

        # model 3: replace sigmoid with relu
        if mode == 3:
            self.model = Model_3(self.hidden_size)

        # model 4: add one extra fully connected layer
        if mode == 4:
            self.model = Model_4(self.hidden_size)

        # model 5: utilize dropout
        if mode == 5:
            self.model = Model_5(self.hidden_size)

    def forward(self, x):
        if self.mode == 1:
            # summary(self.model, (28, 28))
            x = x.view(-1, 28* 28) # Need to be flatten
            x = self.model(x)
        if self.mode in [2, 3, 4, 5]:
            x = self.model(x)
        # ======================================================================
        # Define softmax layer, use the features.
        # ----------------- YOUR CODE HERE ----------------------
        
        logits = nn.LogSoftmax()(x) # LogSoftMax!!

        return logits

# summary(self.model, (1, 28, 28))