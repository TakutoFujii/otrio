import random
import numpy as np
import logging

PIECES = 3
BOARD_SIZE = 3
kouho=[]

player_num, my_turn = tuple(map(int, input().strip().split(' ')))

print (my_turn)


board = np.array(list (map(int, input().strip().split(' '))))
piece = np.zeros(3)
print(0)

def reachCheck(check_num):
    global kouho
    for i in range(BOARD_SIZE):
        for j in range (BOARD_SIZE):
            if board3d[i][j][0] ==check_num:
                if board3d[i][j][1] ==check_num and board3d[i][j][2] ==0:
                    kouho.append([i,j,2])
                elif board3d[i][j][2] ==check_num and board3d[i][j][1] ==0:
                    kouho.append([i,j,1] )
            if board3d[0][i][j] ==check_num:
                if board3d[1][i][j] ==check_num and board3d [2][i][j]==0:
                    kouho.append( [2,i,j])
                elif board3d[2][i][j] ==check_num and board3d[1][i][j] ==0:
                    kouho.append( [1,i,j])
            if board3d[j][0][i] ==check_num:
                if board3d[j][1][i] ==check_num and board3d[j][2][i]==0:
                    kouho.append( [j,2,i] )
                elif board3d [j][2][i] ==check_num and board3d[j][1][i] ==0:
                    kouho.append( [j,1,i])

    for i in range (BOARD_SIZE):
        if board3d[1][1][i] ==check_num:
            if board3d[0][2][i] ==check_num and board3d[2][0][i]==0:
                kouho.append( [2,0,i])
            elif board3d[2][0][i] ==check_num and board3d[0][2][i]==0:
                kouho.append( [0,2,i])
            if board3d[0][0][i] ==check_num and board3d[2][2][i] ==0:
                kouho.append( [2,2,i] )
            elif board3d[2][2][i] ==check_num and board3d[0][0][i]==0:
                kouho.append( [0,0,i] )
        if board3d[1][i][1] ==check_num:
            if board3d[0][i][2] ==check_num and board3d[2][i][0] ==0:
                kouho.append( [2,i,0])
            elif board3d[2][i][0] ==check_num and board3d[0][i][2] ==0:
                kouho.append( [0,i,2])
            if board3d[0][i][0] ==check_num and board3d[2][i][2] ==0:
                kouho.append( [2,i,2])
            elif board3d[2][i][2] ==check_num and board3d[0][i][0] ==0:
                kouho.append( [0,i,0])
        if board3d[i][1][1] ==check_num:
            if board3d[i][0][2] ==check_num and board3d[i][2][0]==0:
                kouho.append( [i,2,0])
            elif board3d[i][2][0] ==check_num and board3d[i][0][2] ==0:
                kouho.append( [i,0,2])
            if board3d[i][0][0] ==check_num and board3d[i][2][2] ==0:
                kouho.append([i,2,2])
            elif board3d[i][2][2] ==check_num and board3d[i][0][0] ==0:
                kouho.append( [i,0,0] )

    if board3d[0][0][2] ==check_num:
        if board3d[1][1][1] ==check_num and board3d[2][2][0] ==0:
            kouho.append( [2,2,0] )
        elif board3d[2][2][0] ==check_num and board3d[1][1][1] ==0:
            kouho.append( [1,1,1] )
    if board3d[2][0][2] ==check_num:
        if board3d[1][1][1] ==check_num and board3d[0][2][0] ==0:
            kouho.append( [0,2,0] )
        elif board3d[0][2][0] ==check_num and board3d[1][1][1] ==0:
            kouho.append( [1,1,1])
    if board3d[2][2][2] ==check_num:
        if board3d[1][1][1] ==check_num and board3d[0][0][0] ==0:
            kouho.append( [0,0,0] )
        elif board3d[0][0][0] ==check_num and board3d[1][1][1] ==0:
            kouho.append( [1,1,1])
    if board3d[0][2][2] ==check_num:
        if board3d[1][1][1] ==check_num and board3d[2][0][0] ==0:
            kouho.append( [2,0,0] )
        elif board3d[2][0][0] ==check_num and board3d[1][1][1] ==0:
            kouho.append( [1,1,1])

def pointDecide():
    n=my_turn
    for i in range(2) :
        reachCheck(n)
        n+=1
        if n>player_num:
            n=1
    k=0
    for i in kouho:
        leftPiece=kouho[k][2]
        k+=1
        if piece[leftPiece] <PIECES:
            piece[leftPiece] +=1
            return kouho[k-1]
    if board3d[1][1][1] ==0 and piece[1]<PIECES:
        piece[1]+=1
        return [1,1,1]

    empty = [i for i, b in enumerate((board == 0).tolist()) if b]
    valid = []
    for e in empty:
        if piece [e % 3] < PIECES:
            valid.append(e)
    if len(valid)==0:
        return [-1,-1,-1]
    move = random.choice(valid)
    piece[move % 3] += 1
    return [(move//3**2) % 3, (move//3) % 3, move % 3]

try:
    while True:
        get = np.array(list(map(int, input().strip().split(' '))))
        turn = get[0]
        board = get[1:]
        #board = np.array(list (map(int, input().strip() .split(' '))))
        #empty = [i for i, b in enumerate((board == 0).tolisto) if b]
        board3d = board.reshape([3, 3, 3] )
        #print (board3d)
        kouho.clear()
        hand=pointDecide()
        print (hand[0] , hand[1] ,hand[2] )
except EOFError:
    pass
