"""

"""
import numpy as np

# data I/O
data = open('input.txt', 'r').read()  # 应该是简单的纯文本文件  英文文本
chars = list(set(data))               #set 是python中的无序的不重复的元素集合
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}#将文本中的字符串转化成字典


# 下面两段是参数的初始化
# 超参
hidden_size = 100  # 隐藏层神经元的大小，也就是隐藏层神经元的维度。hidden的纬度
seq_length = 25  # number of steps to unroll the RNN for  RNN步骤展开的网络数  每次训练的文本的长度是25
learning_rate = 1e-1

# 模型的参数
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # 输入层到隐藏层的矩阵
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏层到隐藏层的矩阵
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # 隐藏层到输出层矩阵
bh = np.zeros((hidden_size, 1))  # hidden bias得到隐藏层是由上一个隐藏层和输入层共同决定的，所以用一个bias就可以了
by = np.zeros((vocab_size, 1))  # output bias


def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass正向传播的过程
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1  #这两句话就是把字母处理成one hot 向量的形式
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state   h的状态结果，更新了内部的h
        ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars    y的状态也得到了
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars  softmax的前置工作
        loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss) log函数里面的值越是靠近1，loss越是小
    # backward pass: compute gradients going backwards反向传播的过程
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why) #zeros_like就是形状和你一样，但是我的值是0
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # backprop into y. softmax的梯度下降
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

#主循环部分
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0:  #循环到头了，或者第一次迭代。n是代数
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data  这个标签可以理解成一个游标 =0就是从文本的开头开始
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]#序列的p到p+seq_length个  这种写法是构造one-hot向量
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt,))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)#得到的hprey是25步以后的h。参数hprev是第一步的hprev，一般初始化为0。在这里是初始化为100纬度的0向量
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0:
        print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    p += seq_length  # move data pointer
    n += 1  # iteration counter