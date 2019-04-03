import random
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np

player_num=4

DRAW = 255
PIECES = 3

BOARD_SIZE = 3
BOARD_LEN = BOARD_SIZE**3

kouho=[]

def reachCheck(check_num,board3d):
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

def pointDecide(turn):
    n=turn
    board3d=b.board.reshape([BOARD_SIZE, BOARD_SIZE, BOARD_SIZE])
    for i in range(2) :
        reachCheck(n,board3d)
        n+=1
        if n>player_num:
            n=1
    k=0
    for i in kouho:
        leftPiece=kouho[k][2]
        k+=1
        if b.piece[turn][leftPiece] <PIECES:
            b.piece[turn][leftPiece] +=1
            return kouho[k-1]
    if board3d[1][1][1] ==0 and b.piece[turn][1]<PIECES:
        b.piece[turn][1]+=1
        return [1,1,1]

    empty = [i for i, bo in enumerate((b.board == 0).tolist()) if bo]
    valid = []
    for e in empty:
        if b.piece[turn][e % 3] < PIECES:
            valid.append(e)
    if len(valid)==0:
        return [-1,-1,-1]
    move = random.choice(valid)
    b.piece[turn][move % 3] += 1
    return [(move//3**2) % 3, (move//3) % 3, move % 3]


class Board():
    def reset(self,player_num):
        self.board=np.array([0]*27,dtype=np.float32)
        self.piece=np.zeros((player_num, 3))
        self.winner=None
        self.missed=False
        self.done=False

    def forward(self,board, order, player_idx):
        board3d = self.board.reshape([BOARD_SIZE, BOARD_SIZE, BOARD_SIZE])
        if board3d[order[0], order[1], order[2]] == 0 and order[0]!=-1:
            board3d[order[0], order[1], order[2]] = player_idx
        else:
            self.missed=True
            self.done=True

    def judge(self):
        board3d = self.board.reshape([BOARD_SIZE, BOARD_SIZE, BOARD_SIZE])
        max_idx = board3d.max()
        res = 0
        for idx in range(1, int(max_idx)+1):
            faces = []
            f1 = []
            f2 = []
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if all(board3d[i, j, :] == idx) or all(board3d[:, i, j] == idx) or all(board3d[i, :, j] == idx):
                        # tate yoko
                        #assert not(res != 0 and res !=
                        #           idx), 'wrong judge tate yoko'
                        res = idx
                    f1.append(board3d[i, j, i])
                    f2.append(board3d[2-i, j, i])
                faces.append(board3d[i, :, :])
                faces.append(board3d[:, i, :])
                faces.append(board3d[:, :, i])
            faces.append(np.array(f1).reshape([BOARD_SIZE, BOARD_SIZE]))
            faces.append(np.array(f2).reshape([BOARD_SIZE, BOARD_SIZE]))
            for face in faces:
                if all(np.diag(face) == idx) or all(np.diag(np.fliplr(face)) == idx):
                    # across
                    #assert not(res != 0 and res != idx), 'wrong judge across'
                    res = idx
        if res == 0:
            val_len = [len(self.valid_move(i)) == 0 for i in range(player_num)]
            if all(val_len):
                res = DRAW
        self.winner=res

    def valid_move(self, num):
        board=self.board.copy()
        board3d, piece = self.board23d(board, num)
        av_piece = piece <= PIECES
        valid = []
        for i, av in enumerate(av_piece):
            if av:
                b = board3d[:, :, i]
                for j, row in enumerate(b):
                    for k, pix in enumerate(row):
                        if pix == 0:
                            valid.append([j, k, i])

        return valid

    def board23d(self,board, num):
        board3d = b.board.reshape([BOARD_SIZE, BOARD_SIZE, BOARD_SIZE])
        bo = (board3d == num).astype(int)
        piece = bo.sum((0, 1))
        return board3d, piece

class RandomActor:
    def __init__(self,board,piece):
        self.board=board
        self.piece=piece
        self.random_count=0
    def random_action_func(self):
        self.random_count+=1
        valid = b.valid_move(1)
        if len(valid) == 0:
            return [-1,-1,-1]
        move = random.choice(valid)
        return move

    def random_player_v2(self,turn):
        hand=pointDecide(turn)
        kouho.clear()
        return hand

class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=81):
        super().__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_hidden_channels),
            l3=L.Linear(n_hidden_channels, n_actions))
    def __call__(self, x, test=False):
        #-1を扱うのでleaky_reluとした
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = F.leaky_relu(self.l2(h))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))

b=Board()
b.reset(player_num)
print(b)
ra=RandomActor(b.board,b.piece)

obs_size = 27
n_actions = 27
# Q-functionとオプティマイザーのセットアップ
q_func = QFunction(obs_size, n_actions)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
# 報酬の割引率
gamma = 0.95
# Epsilon-greedyを使ってたまに冒険。50000ステップでend_epsilonとなる
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000, random_action_func=ra.random_action_func)
# Experience ReplayというDQNで用いる学習手法で使うバッファ
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
# Agentの生成（replay_buffer等を共有する2つ）
agent_p = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500,target_update_interval=100,
    update_interval=1)

n_episodes=200
miss=0
win=0
draw=0

for i in range(1,n_episodes+1):
    b.reset(player_num)
    reward=0
    turn=np.random.choice([0,1,2,3])
    last_state = None
    while not b.done:
        for j in range(player_num):
            if turn==0:
                action=agent_p.act_and_train(b.board.copy(),reward)
                print(action)
            else:
                action=ra.random_player_v2(turn)

            b.forward(b.board.copy(),action,turn+1)
            b.judge()
            turn+=1
            if turn>=player_num:
                turn=0

            if b.done==True:
                if b.winner==1:
                    reward=1
                    win+=1
                elif b.winner==DRAW:
                    draw+=1
                else:
                    reward=-1
                if b.missed is True:
                    miss+=1
                agent_p.stop_episode_and_train(b.board.copy(),reward,True)
            #else:
                #last_state = b.board.copy()

    if i%100==0:
        print("episode:", i, " / rnd:", ra.random_count, " / miss:", miss, " / win:", win, " / draw:", draw, " / statistics:", agent_p1.get_statistics(), " / epsilon:", agent_p1.explorer.epsilon)
        miss=0
        win=0
        draw=0
        ra.random_count=0

print("Training finished")
