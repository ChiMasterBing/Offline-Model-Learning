# Simulation of the Othello environment

def getStrip(x, y, incX, incY):
    res = []
    while 0 <= x < 8 and 0 <= y < 8:
        res.append(DTS[x, y])
        x += incX; y += incY
    return res

def getPossibleMovesDots(board, token):
    global opponent
    foe = opponent[token]
    res = set()
    for i in range(64):
        if board[i] == '.':
            for strp in sides[i]:
                added = False
                for j, c in enumerate(strp[1:]):
                    if board[c] == token and j >= 1:
                        res.add(i)
                        added = True
                        break
                    elif board[c] == foe:
                        continue
                    else:
                        break
                if added: break
    return res

def playMove(board, char, position):
    board = [*board]
    board[position] = char
    for strp in sides[position]:
        stck = []
        for c in strp[1:]:
            stck.append(c)
            if board[c] == char:
                while stck:
                    tmp = stck.pop()
                    board[tmp] = char
                break
            elif board[c] == opponent[char]:
                continue
            else:
                break
                
    return ''.join(board)

def playMoveTpl(board, char, position, bad = False):
    pM = getPossibleMovesDots(board, char)
    if position not in pM:
        if not bad:
            return playMoveTpl(board, opponent[char], position, True)
        return False

    board = [*board]
    board[position] = char
    for strp in sides[position]:
        stck = []
        for c in strp[1:]:
            stck.append(c)
            if board[c] == char:
                while stck:
                    tmp = stck.pop()
                    board[tmp] = char
                break
            elif board[c] == opponent[char]:
                continue
            else:
                break
                
    return (''.join(board), opponent[char])

def setGlobals():
    #position lookup tables
    global STATS 
    STATS = {}

    global opponent, nbrs, sides, DTS, STD, LTN, cutSides
    opponent = {}
    opponent['x'] = 'o'; opponent['o'] = 'x'

    DTS, STD = {}, {}
    for i in range(8): #row
        for j in range(8): #col
            DTS[(i, j)] = i*8+j
            STD[i*8+j] = (i, j)

    nbrs, sides, cutSides = {}, {}, {}
          
    dx = [-1, 1, -1, 1, 0, 0, -1, 1]
    dy = [-1, 1, 0, 0, -1, 1, 1, -1]
    for i in range(8): #row
        for j in range(8): #col
            nbrs[DTS[(i, j)]] = []
            sides[DTS[(i, j)]] = []
            cutSides[DTS[(i, j)]] = []
            for cx, cy in zip(dx, dy):
                nx = i+cx; ny = j+cy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    nbrs[DTS[(i, j)]].append(DTS[(nx, ny)])
                else:
                    nbrs[DTS[(i, j)]].append(-1)
                strip = getStrip(i, j, cx, cy)
                sides[DTS[(i, j)]].append(strip)
                if len(strip[1:]) > 1:
                    cutSides[DTS[(i, j)]].append([(i, j) for i, j in enumerate(strip[1:])])

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    LTN = {l:i+1 for i, l in enumerate(letters)}

    global edgeStrips
    edgeStrips = []
    edgeStrips.append((0, 8, 1))
    edgeStrips.append((56, 64, 1))
    edgeStrips.append((0, 57, 8))
    edgeStrips.append((7, 64, 8))


if __name__ == "__main__" :
    setGlobals()
    startBoard = '.'*27 + "ox......xo" + '.'*27
    print(getPossibleMovesDots(startBoard, 'x'))