brd = '...................xxx....xoox....xxoox...xxxoo....xxx.....xo.x.'
tkn = 'o'

#Runs a sample prediction on (brd, tkn) above

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional

class NN(nn.Module): #simple CNN
    def __init__(self):
        super().__init__()
        
        self.C1 = nn.Conv2d(2, 64, 3, padding=1)
        self.C2 = nn.Conv2d(64, 128, 3, padding=1)
        self.P1 = nn.MaxPool2d(2, 2)

        #self.C3 = nn.Conv2d(128, 256, 3)

        #padding?
        self.flatten = nn.Flatten(-3, -1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4*4*128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),   
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = functional.relu(self.C1(x))
        x = functional.relu(self.C2(x))
        x = self.P1(x)

        x = self.flatten(x)

        logits = self.linear_relu_stack(x)
        return logits


class agent:
    
    func = None
    device = None
    opt = None
    lossFunc = None

    LR = 0.0001
    PV = 0.9
    GREEDY = 0.1

    def __init__(self):
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using {self.device} device")
        self.func = NN().to(self.device)
        self.opt = torch.optim.Adam(self.func.parameters(), lr = self.LR)

        self.lossFunc = nn.MSELoss()

        self.softmaxCache = {}

        self.rng64 = [i for i in range(64)]

        self.func.load_state_dict(torch.load("50kOffline.pth"))

    def scoot(self, brd, tkn, g):
        g = Variable(torch.tensor(g), requires_grad = False).to(self.device)
        
        pred = self.predict(brd, tkn)

        loss = self.lossFunc(pred, g)

        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        return loss.item()
    
    def constructTensor(self, brd, tkn):
        self.lastBrd = brd #for caching purposes
        self.lastTkn = tkn

        # 8 x 8 x 2
        # 1 --> current player's tkns
        # 2 --> opponent's tkns
        opp = 'x' if tkn == 'o' else 'o'

        l1 = [[1.0 if brd[i*8+j] == tkn else 0.0 for j in range(8)] for i in range(8)]
        l2 = [[1.0 if brd[i*8+j] == opp else 0.0 for j in range(8)] for i in range(8)]

        f = [l1, l2]

        f = Variable(torch.tensor(f), requires_grad = True).to(self.device)

        return f


    def predict(self, f):
        pred = self.func(f)
        return pred
    
agt = agent()
f = agt.constructTensor(brd, tkn)
res = agt.predict(f).tolist()
print(res)
    
    