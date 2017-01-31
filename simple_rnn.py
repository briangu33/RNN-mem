"""
Vanilla RNN. Based on code written by Andrej Karpathy (@karpathy)
"""

import numpy as np

class Simple_RNN:
    
    def __init__(self, hidden_size, seq_length):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
    def lossFun(self, inputs, targets, hprev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        # forward pass
        for t in xrange(len(inputs)):
            xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
            ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
            loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(xrange(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
    
    def sample(self, n, h = None, savefile = None, seed_ix = 0):
        """ 
        sample a sequence of integers from the model 
        h is memory state, seed_ix is seed letter for first time step
        """
        if h == None:
            h = np.zeros((self.hidden_size, 1))
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        hs = np.empty((n, self.hidden_size, 1))
        for t in xrange(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
            hs[t] = h
        hs = np.reshape(hs, (n, self.hidden_size)).tolist()
        txt = self.ixes_to_chars(ixes)
        to_save = [[char] + h for char, h in zip(txt, hs)]
        if savefile != None:
            np.savetxt(savefile, hs, delimiter=',')
        return txt, hs
    
    def ixes_to_chars(self, ixes):
        txt = ''.join(self.ix_to_char[ix] for ix in ixes)
        # print '----\n%s\n----' % (txt, )
        return txt
    
    def train(self, training_data, iterations):
        # data IO
        # Available training data:
        #  - char_printer_input.txt
        #  - char_count_input.txt
        #  - 2_rule_grammar_10.txt
        #  - 2_rule_grammar_30.txt
        #  - 2_rule_grammar_50.txt
        #  - 2_rule_grammar_100.txt
        data = open(training_data, 'r').read() # should be simple plain text file
        self.chars = sorted(list(set(data)))
        self.data_size, self.vocab_size = len(data), len(self.chars)
        print 'data has %d characters, %d unique.' % (self.data_size, self.vocab_size)
        self.char_to_ix = { ch:i for i,ch in zip(range(self.vocab_size), self.chars) }
        self.ix_to_char = { i:ch for i,ch in zip(range(self.vocab_size), self.chars) }
        
        # initialize parameters
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.1 # input to hidden
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.1 # hidden to hidden
        self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.1 # hidden to output
        self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
        self.by = np.zeros((self.vocab_size, 1)) # output bias
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad
        smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length # loss at iteration 0
        
        # hyperparameters
        learning_rate = 1.
        
        n, p = 0, 0
        while n < iterations:
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p+self.seq_length+1 >= len(data) or n == 0: 
                hprev = np.zeros((self.hidden_size,1)) # reset RNN memory
                p = 0 # go from start of data
            inputs = [self.char_to_ix[ch] for ch in data[p:p+self.seq_length]]
            targets = [self.char_to_ix[ch] for ch in data[p+1:p+self.seq_length+1]]

            # sample from the model now and then
            if n % 20000 == 0:
                self.sample(200, hprev)[0]

            # forward seq_length characters through the net and fetch gradient
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if n % 1000 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
                                                                        [dWxh, dWhh, dWhy, dbh, dby], 
                                                                        [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

            p += self.seq_length # move data pointer
            n += 1 # iteration counter 