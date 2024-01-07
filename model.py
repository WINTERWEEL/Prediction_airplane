import torch.nn as nn

class MultiLayer(nn.Module):
    def __init__(self,args):
        super(MultiLayer,self).__init__()
        self.dropout = args.dropout
        self.input_size = args.num_feature
        self.output_size = 1
        self.hidden_size = args.hidden_size
        self.num_hidden_layers = args.num_layers-2
        self.batnrm = args.batch_norm
        self.relu = nn.ReLU()

        self.input_layer = nn.Linear(self.input_size,self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size,1)
        self.batchnorm = nn.BatchNorm1d(self.hidden_size)
        if args.num_layers<0:
            raise ValueError("Sorry, num_layers can not be lower than 2")
        else:
            self.hidden_layers = nn.Sequential()
            for i in range(self.num_hidden_layers):
                self.hidden_layers.add_module('hidden'+str(i+1),nn.Linear(self.hidden_size,self.hidden_size))
                self.hidden_layers.add_module('relu'+str(i+1),nn.ReLU())
                if self.dropout>0:
                    self.hidden_layers.add_module('relu'+str(i+1), nn.Dropout(p=self.dropout))

    def forward(self,x):
        # print(x.shape)
        x = self.input_layer(x)
        if self.batnrm:
            x = self.batchnorm(x)
        x = self.relu(x)
        x = self.hidden_layers(x)
        out= self.output_layer(x)
        return out



