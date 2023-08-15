import othelloDriver as OD
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional
import random

# Offline learning agent

# The model to be loaded to continue training. Leave this value empty or false to not load any file.
LOAD_MODEL = "" #"in.pth"

# The model to save to. Saves happen every 500 episodes. Leave empty or false for no saving.
SAVE_MODEL = "" #"out.pth"

# The file to write stats to. Leave empty or false for no saving.
STATS_FILE = "" # output.txt

# How many episodes to train for
EPISODE_COUNT = 5000

# Distribution to offline train on. "Human" for expert human games. Anything else is random distribution.
DISTRIBUTION = ""

class NN(nn.Module): #simple CNN
    def __init__(self):
        super().__init__()
        
        self.C1 = nn.Conv2d(2, 64, 3, padding=1)
        self.C2 = nn.Conv2d(64, 128, 3, padding=1)
        self.P1 = nn.MaxPool2d(2, 2)

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

        if LOAD_MODEL:
            self.func.load_state_dict(torch.load(LOAD_MODEL))        

    def scoot(self, brd, tkn, g):
        g = Variable(torch.tensor(g), requires_grad = False).to(self.device)
        
        pred = self.predict(brd, tkn)

        loss = self.lossFunc(pred, g)

        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        return loss.item()
    
    def predict(self, brd, tkn):
        # 8 x 8 x 2
        # 1 --> current player's tkns
        # 2 --> opponent's tkns
        opp = 'x' if tkn == 'o' else 'o'

        l1 = [[1.0 if brd[i*8+j] == tkn else 0.0 for j in range(8)] for i in range(8)]
        l2 = [[1.0 if brd[i*8+j] == opp else 0.0 for j in range(8)] for i in range(8)]

        f = [l1, l2]


        f = Variable(torch.tensor(f), requires_grad = True).to(self.device)

        pred = self.func(f)

        return pred
    
    def softmax(self, vec):
        sm = sum(vec)
        return [v/sm for v in vec]

    def simulateMove(self, brd, tkn, egreed = True):
        #print(brd, tkn)
        pred = self.predict(brd, tkn).tolist()
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

def train(f, g, agt):
    
    f = Variable(torch.tensor(f), requires_grad = True) #batch, dim
    g = Variable(torch.tensor(g), requires_grad = False)

    f = f.to(agt.device)
    g = g.to(agt.device)

    pred = agt.func(f)

    loss = agt.lossFunc(pred, g)

    loss.backward()
    agt.opt.step()
    agt.opt.zero_grad()

    return loss.item()

def simulateEpisode(agt, epIndex):
    global avgd
    batch = []
    depth = 0
    global episodes
    epIndex %= len(episodes)
    episode = episodes[epIndex]
    

    for info in episode:
        if type(info) == int: # A
            picked = info

            pred = agt.predict(brd, tkn).tolist()

            diff = 1 - pred[picked]
            arr = [(1-pred[pos])*pred[pos] if pos != picked else 0 for pos in range(64)]
            sm = sum(arr)
        
            for pos in range(64):
                if pos == picked:
                    pred[pos] = 1.0
                else:
                    pred[pos] -= (arr[pos]/sm) * diff
            
            batch.append((brd, tkn, pred))
        else: # S
            depth += 1
            brd, tkn = info
        
    avgd += depth

    f, g = [], []
    for tpl in batch:
        brd, tkn, ans = tpl
        opp = 'x' if tkn == 'o' else 'o'
        l1 = [[1.0 if brd[i*8+j] == tkn else 0.0 for j in range(8)] for i in range(8)]
        l2 = [[1.0 if brd[i*8+j] == opp else 0.0 for j in range(8)] for i in range(8)]

        f.append([l1, l2])
        g.append(ans)

    lss = train(f, g, agt)

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

            agtChoice, pred = agt.simulateMove(brd, tkn, False)

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

            picked = random.choice([*mvs])

            brd = OD.playMove(brd, tkn, picked)
            tkn = OD.opponent[tkn]
        
        if trial%5 == 0:
            print('$', end='', flush=True)
    
    if outfile:
        outfile.write(f"Iterations: {iterations}\n")
        outfile.write(f"Rounded Error: {roundedError/100}\n")
        outfile.write(f"MSE Error: {squaredError/100}\n")
        outfile.write(f"Random Goofs: {randomGoof/100}\n")
        outfile.write(f"Selection Goofs: {selectionGoof/100}\n")
        outfile.write(f"Avg Depth: {totalDepth/100}\n")
        outfile.flush()

    print()
    print(f"Iterations: {iterations}")
    print(f"Rounded Error: {roundedError/100}")
    print(f"MSE Error: {squaredError/100}")
    print(f"Random Goofs: {randomGoof/100}")
    print(f"Selection Goofs: {selectionGoof/100}")
    print(f"Avg Depth: {totalDepth/100}")

    agt.func.train()

def generateEpisodes(count):
    global episodes
    episodes = []

    print("Generating Episodes")

    for i in range(count):
        episode = []
        brd = '.'*27 + "ox......xo" + '.'*27
        tkn = 'x'

        depth = 0
        while True:
            depth += 1
            mvs = OD.getPossibleMovesDots(brd, tkn)
            if not mvs:
                break
            episode.append((brd, tkn))
            
            mvs = sorted([*mvs])
            picked = random.choice([*mvs][:3])
            

            episode.append(picked)
            brd = OD.playMove(brd, tkn, picked)
            tkn = OD.opponent[tkn]

        episodes.append(episode)

        if i%1000 == 0:
            print('#', end='', flush=True)
    print()
    


if __name__ == "__main__":
    global outfile
    if STATS_FILE:
        outfile = open(STATS_FILE, 'w')
    else:
        outfile = False

    global avgd
    avgd = 0
    agt = agent()
    OD.setGlobals()

    if DISTRIBUTION == "human":
        import Wthor
        global episodes
        episodes = Wthor.generateEpisodes(EPISODE_COUNT + 100)
    else:
        generateEpisodes(EPISODE_COUNT + 100)

    for i in range(EPISODE_COUNT + 1):
        if i%500 == 0:
            if SAVE_MODEL:
                torch.save(agt.func.state_dict(), SAVE_MODEL)
        if i%1000 == 0:
            fullTest(agt, i)
        if i%10 == 0:
            print(f"*", end="", flush=True)
        simulateEpisode(agt, i)
        
