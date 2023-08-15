import othelloDriver as OD

# .wtb file parser - reads the human games and processes them into agent friendly format

def wtbToMove(i):
    if i == 0:
        return -1
    y = (i//10)-1
    x = (i%10)-1
    return y*8+x

def checkValidMove(i):
    return i <= 88

def playGame(moves):
    episode = []

    board = '.'*27 + "ox......xo" + '.'*27
    token = 'x'

    for move in moves:
        if move == -1:
            return episode
        episode.append((board, token))
        
        episode.append(move)

        tpl = OD.playMoveTpl(board, token, move)
        if not tpl:
            return False
        board, token = tpl

    return episode

def parseGame(bytes):
    if not (0 <= bytes[6] <= 64 and 0 <= bytes[7] <= 64):
        return False 
    moves = []
    for i in bytes[8:]:
        if not checkValidMove(i):
            return False
        moves.append(wtbToMove(i))
    episode = playGame(moves)
    return episode

def generateEpisodes(cnt):
    print("Parsing Wthor files...")
    episodes = []
    for year in range(2016, 1977, -1):
        with open(f"HumanGames/WTH_{year}.wtb", mode="rb") as f:
            contents = f.read()
        game = 0
        while True:
            if 16+game*68+68 >= len(contents):
                break
            if game%50 == 0:
                print("*", end="", flush=True)
            if not (episode:=parseGame(contents[16+game*68:16+game*68+68])):
                break
            episodes.append(episode)
            if len(episodes) >= cnt:

                return episodes
            game += 1