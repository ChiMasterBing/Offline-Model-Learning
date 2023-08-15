import othelloDriver as OD
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional
import random

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
        #x = functional.relu(self.C3(x))
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

        self.func.load_state_dict(torch.load("1010Human.pth"))
        
        #self.opt = torch.optim.SGD(self.func.parameters(), lr = self.LR, momentum = self.PV)
        

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

        # 8 x 8 x 3
        # Similar to AlphaGoZero
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
    
    def softmax(self, vec):
        if (self.lastBrd, self.lastTkn) in self.softmaxCache:
            return self.softmaxCache[(self.lastBrd, self.lastTkn)]
        sm = sum(vec)
        res = [v/sm for v in vec]
        self.softmaxCache[(self.lastBrd, self.lastTkn)] = res
        return res
    
    def simulateMove(self, brd, tkn, pred=False, egreed=True):
        if pred == False: 
            pred = self.predict(self.constructTensor(brd, tkn)).tolist()
        if egreed:
            eps = random.random()
        else:
            eps = 1
        if eps < self.GREEDY:
            #do e-greedy move
            return (random.randint(0, 63), pred)
        else:
            # ^ 64x1 vector, probabilities
            probs = self.softmax(pred)
            return (random.choices([i for i in range(64)], weights=probs)[0], pred)

    def simulateMoveTensor(self, f, pred=False):
        if pred == False: 
            pred = self.predict(f).tolist()

        eps = random.random()
        if eps < self.GREEDY:
            #do e-greedy move
            return (random.randint(0, 63), pred)
        else:
            # ^ 64x1 vector, probabilities
            probs = self.softmax(pred)
            return (random.choices(self.rng64, weights=probs)[0], pred)



def train(f, g, agt):

    f = f.to(agt.device)
    g = g.to(agt.device)

    # print(f.shape)
    # print(g.shape)

    pred = agt.func(f)

    loss = agt.lossFunc(pred, g)

    loss.backward()
    agt.opt.step()
    agt.opt.zero_grad()

    return loss.item()

def test(agt):
    brd = '.'*27 + "ox......xo" + '.'*27
    tkn = 'x'

    goofs = 0
    while True:
        mvs = OD.getPossibleMovesDots(brd, tkn)
        
        if not mvs:
            print(brd)
            break
                
        picked, pred = agt.simulateMove(brd, tkn)
        
        #print(picked)

        if picked not in mvs:
            goofs += 1
        else:
            brd = OD.playMove(brd, tkn, picked)
            tkn = OD.opponent[tkn]

    print("Goofs", goofs)

def fullTest(agt, iterations):
    agt.func.eval()

    roundedError = 0
    squaredError = 0
    randomGoof = 0
    selectionGoof = 0
    totalDepth = 0
    print()
    for trial in range(100):
        brd = '.'*27 + "ox......xo" + '.'*27
        tkn = 'x'

        while True:
            mvs = OD.getPossibleMovesDots(brd, tkn)
            
            if not mvs:
                totalDepth += 64 - brd.count('.')
                break

            agtChoice, pred = agt.simulateMove(brd, tkn, egreed=False)

            ans = [1.0 if mv in mvs else 0.0 for mv in range(64)]
            mx, mxInd = -1, -1
            for pos in range(64):
                roundedChoice = round(pred[pos])
                if int(roundedChoice) != int(ans[pos]):
                    roundedError += 1
                squaredError += (pred[pos] - ans[pos]) ** 2

                if pred[pos] > mx:
                    mx = pred[pos]
                    mxInd = pos
            
            if int(ans[agtChoice]) != 1:
                randomGoof += 1
            
            if int(ans[mxInd]) != 1:
                selectionGoof += 1

            print(sum(pred))
            print("\n\n")

            picked = random.choice([*mvs])

            brd = OD.playMove(brd, tkn, picked)
            tkn = OD.opponent[tkn]
        
        if trial%5 == 0:
            print('$', end='', flush=True)
    print()
    outfile.write(f"Iterations: {iterations}\n")
    outfile.write(f"Rounded Error: {roundedError/100}\n")
    outfile.write(f"MSE Error: {squaredError/100}\n")
    outfile.write(f"Random Goofs: {randomGoof/100}\n")
    outfile.write(f"Selection Goofs: {selectionGoof/100}\n")
    outfile.write(f"Avg Depth: {totalDepth/100}\n")
    outfile.flush()

    print(f"Iterations: {iterations}")
    print(f"Rounded Error: {roundedError/100}")
    print(f"MSE Error: {squaredError/100}")
    print(f"Random Goofs: {randomGoof/100}")
    print(f"Selection Goofs: {selectionGoof/100}")
    print(f"Avg Depth: {totalDepth/100}")

    agt.func.train()

def simulateEpisode(agt):
    global avgd
    brd = '.'*27 + "ox......xo" + '.'*27
    tkn = 'x'

    batch = []

    depth = 0
    while True:
        depth += 1
        mvs = OD.getPossibleMovesDots(brd, tkn)
        
        if not mvs:
            break

        f = agt.constructTensor(brd, tkn)
        picked, pred = agt.simulateMoveTensor(f)
        
        while picked not in mvs:     
            if picked != -1:
                pred[picked] = 0
            picked, __ = agt.simulateMoveTensor(f, pred=pred)

        
        pred[picked] = 1

        batch.append((f, pred))

        brd = OD.playMove(brd, tkn, picked)
        tkn = OD.opponent[tkn]
        
    avgd += depth

    f, g = [], []
    for tpl in batch:
        fe, ans = tpl
        f.append(fe)
        g.append(ans)

    f = torch.stack(f, 0)
    g = Variable(torch.tensor(g), requires_grad = False)

    lss = train(f, g, agt)


if __name__ == "__main__":
    outfile = open("1010Human.txt", 'w')

    global avgd
    avgd = 0
    agt = agent()
    OD.setGlobals()
    for i in range(40001):
        if i%10 == 0:
            print(f"*", end="", flush=True)
        # if i%250 == 0:
        #     print(f"Test {i}")
        #     test(agt)
        #     print("Average Depth", avgd/100)
        #     avgd = 0
            
        if i%1000 == 0:
            fullTest(agt, i)
            #if i != 0: torch.save(agt.func.state_dict(), "wthor.pth")

        simulateEpisode(agt)
        





