import theano.tensor as TT

# Needs:
#   model.x
#   model.layers[i].a, model.layers[i].s


class Optimizer:
    def __init__(model, loss, diag=True):
        self.model = model
        self.loss = loss
        self.diag = diag

    def calcKFAC(grad_vec, damp):
        self.grads = []
        # self.acts = [TT.concatenate([self.model.x, TT.ones((self.model.x.shape[0], 1))], axis=1)]
        self.acts = [self.model.x]
        for l in self.model.layers:
            S = TT.grad(self.loss, l.s)
            self.grads.append(S)
            self.acts.append(l.a)

        self.G = []
        self.A = []
        self.F_block = []
        self.F = []

        cnt = TT.cast(self.grads[0].shape[0], theano.config.floatX)
        for i in range(len(self.grads)):
            self.G += [[]]
            self.A += [[]]
            for j in range(len(self.grads)):
                # self.G[-1] += [TT.mean(TT.batched_dot(self.grads[i].dimshuffle(0, 1, 'x'), self.grads[j].dimshuffle(0, 'x', 1)), 0).dimshuffle('x', 0, 1)]
                # self.A[-1] += [TT.mean(TT.batched_dot(self.acts[i].dimshuffle(0, 1, 'x'), self.acts[j].dimshuffle(0, 'x', 1)), 0).dimshuffle('x', 0, 1)]

                # self.G[-1] += [TT.batched_dot(self.grads[i].dimshuffle(0, 1, 'x'), self.grads[j].dimshuffle(0, 'x', 1))]
                # self.A[-1] += [TT.batched_dot(self.acts[i].dimshuffle(0, 1, 'x'), self.acts[j].dimshuffle(0, 'x', 1))]

                self.G[-1] += [self.grads[i].TT.dot(self.grads[j]).dimshuffle('x', 0, 1)/cnt]
                self.A[-1] += [self.acts[i].TT.dot(self.acts[j]).dimshuffle('x', 0, 1)/cnt]

                if self.diag:
                    self.G[-1][-1] *= float(i==j)
                    self.A[-1][-1] *= float(i==j)


        for i in range(len(self.grads)):
            self.F_block += [[]]
            for j in range(len(self.grads)):
                # depends on whether you want to compute the real fisher with this or the kr approximation
                # since numpy-base fast_kron somehow computes 3d tensors faster than theano

                # cblock = fast_kron(self.A[i][j], self.G[i][j])
                cblock = native_kron(self.A[i][j], self.G[i][j])

                cblock = cblock.reshape(cblock.shape[1:], ndim=2)
                self.F_block[i] += [cblock]
            self.F.append(TT.concatenate(self.F_block[-1], axis=1))
        self.F = TT.concatenate(self.F, axis=0)
        self.F = (self.F+self.F.T)/2


        self.Fdamp = self.F+TT.identity_like(self.F)*damp

        # new_grad_vec = theano.tensor.slinalg.solve(self.Fdamp, grad_vec.dimshuffle(0, 'x'))
        new_grad_vec = solve_sym_pos(self.Fdamp, grad_vec)
        # new_grad_vec = gpu_solve(self.Fdamp, grad_vec.dimshuffle(0, 'x'))

        return new_grad_vec