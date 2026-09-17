# cerating the model 
class CharModel(nn.Module):
    def __init__(self,all_chars,num_hidden=256,num_layers=4,drop_prob=0.5,use_gpu=False):

        super().__init__()
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.use_gpu = use_gpu

        self.all_chars = all_chars
        self.decoder = dict(enumerate(all_chars))
        self.encoder = {char:ind for ind,char in decoder.items()}

        self.lstm=nn.LSTM(len(self.all_chars),num_hidden,num_layers,dropout=drop_prob,batch_first=True)
        self.dropout = nn.Dropout(drop_prob)

        self.fc_linear = nn.Linear(num_hidden,len(self.all_chars))

    def forward(self,x,hidden):
        lstm_output, hidden = self.lstm(x,hidden)

        drop_output = self.dropout(lstm_output)

        drop_out = drop_output.contiguous().view(-1,self.num_hidden)

        final_out = self.fc_linear(drop_output)
        return final_out,hidden

    def hidden_state (self,batch_size):
        if self.use_gpu:
            hidden = (torch.zeros(self.num_layers,batch_size,self.num_hidden).cuda(),
                      torch.zeros(self.num_layers,batch_size,self.num_hidden).cuda())
        else:
            hidden = (torch.zeros(self.num_layers,batch_size,self.num_hidden),
                     torch.zeros(self.num_layers,batch_size,self.num_hidden))
        return hidden