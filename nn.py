

from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import data


# Set the random number generators' seeds for consistency
SEED = 9876
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    I do not want to shuffle the dataset. 
    Set shuffle = False
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, _p, trng):
    '''proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.8, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 1.0)'''

    proj = state_before * trng.binomial(state_before.shape,
                                        p=_p, n=1,
                                        dtype=state_before.dtype)
    return proj




def dropout_mask_1D(state, dim, _p, trng):
    return trng.binomial(size=(state.shape[dim],), p=_p, n=1, dtype=state.dtype)




def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options, Wemb_value=None):
    """
    Global (not CNN) parameter. For the embeding and the classifier.
    """

    rng = numpy.random.RandomState(7896)


    params = OrderedDict()

    # embeddings of cue types
    params['CueTemb'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['dim_cueType'])),
                            high = numpy.sqrt(6. / (options['dim_cueType'])),
                            size=(options['n_cueTypes'], options['dim_cueType'])
                            )
                        )
                    ).astype(theano.config.floatX)

    # embeddings of differences of locations
    params['Lemb'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['dim_locDiff'])),
                            high = numpy.sqrt(6. / (options['dim_locDiff'])),
                            size=(options['n_locDiffs'], options['dim_locDiff'])
                            )
                        )
                    ).astype(theano.config.floatX)

    # embeddings of tokens in sentences
    if Wemb_value is None:
        params['Wemb'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['dim_token'])),
                            high = numpy.sqrt(6. / (options['dim_token'])),
                            size=(options['n_words'], options['dim_token'])
                            )
                        )
                    ).astype(theano.config.floatX)
    else:
        params['Wemb'] = (numpy.asarray(Wemb_value)).astype(theano.config.floatX)


    #NN parameters
    params = param_init_cnn(options, params, prefix='cnn')
    params = param_init_lstm(options, params, prefix='lstm')

    # classifier softmax
    params['W30'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['dim_token'] * 2 + options['cnn_n1'] + options['ydim0'])),
                            high = numpy.sqrt(6. / (options['dim_token'] * 2 + options['cnn_n1'] + options['ydim0'])),
                            size=(options['dim_token'] * 2 + options['cnn_n1'], options['ydim0'])
                            )
                        )
                    ).astype(theano.config.floatX)

    params['b30'] = numpy.zeros((options['ydim0'],)).astype(config.floatX)



    params['W31'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['n2'] + options['ydim1'])),
                            high = numpy.sqrt(6. / (options['n2'] + options['ydim1'])),
                            size=(options['n2'], options['ydim1'])
                            )
                        )
                    ).astype(theano.config.floatX)

    params['b31'] = numpy.zeros((options['ydim1'],)).astype(config.floatX)


    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams




def param_init_cnn(options, params, prefix='cnn'):
    """
    Init the CNN parameter:

    :see: init_params
    """

    rng = numpy.random.RandomState(6789)

    #parameters of Convolutional Layer
    W1 = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['dim_token'] + options['cnn_n1'])),
                            high = numpy.sqrt(6. / (options['dim_token'] + options['cnn_n1'])),
                            size=(options['dim_token'], options['cnn_n1'])
                            )
                        )
                    ).astype(theano.config.floatX)
    params[_p(prefix, 'W1')] = W1


    b1 = numpy.zeros((options['cnn_n1'],)).astype(theano.config.floatX)
    params[_p(prefix, 'b1')] = b1


    params[_p(prefix, 'V1')] = rng.normal(scale=0.1, size=(options['cnn_n1'],)).astype(config.floatX)



    #parameters of Hidden Layer
    '''W2 = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(1. / (options['n1'] * 3 + options['n2'])),
                            high = numpy.sqrt(1. / (options['n1'] * 3 + options['n2'])),
                            size=(options['n1'] * 3, options['n2'])
                            )
                        )
                    ).astype(theano.config.floatX)
    params[_p(prefix, 'W2')] = W2

    b2 = numpy.zeros((options['n2'],)).astype(theano.config.floatX)
    params[_p(prefix, 'b2')] = b2'''

    return params


def cnn_layer(tparams, state, mask, options, trng, prefix='cnn'):
 

    #Convolutional Layer
    y1 = tensor.dot(state * mask[:, :, None], tparams[_p(prefix, 'W1')])


    # Attention
    y01 = tensor.tanh(y1) * mask[:, :, None] * tparams[_p(prefix, 'V1')]
    alpha = tensor.nnet.softmax(y01.sum(axis=2))# * mask
    y1 = y1 * alpha[:, :, None]

    conv_out0 = (y1 * mask[:, :, None]).sum(axis=1)

        
    return tensor.tanh(conv_out0)




def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)




def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM_0 parameter:

    :see: init_params
    """

    rng = numpy.random.RandomState(6789)

    params[_p(prefix, 'Wf')] = numpy.concatenate([ortho_weight(options['dim_token']),
                                                  ortho_weight(options['dim_token']),
                                                  ortho_weight(options['dim_token']),
                                                  ortho_weight(options['dim_token'])], axis=1)


    params[_p(prefix, 'Uf')] = numpy.concatenate([ortho_weight(options['dim_token']),
                                                  ortho_weight(options['dim_token']),
                                                  ortho_weight(options['dim_token']),
                                                  ortho_weight(options['dim_token'])], axis=1)

    params[_p(prefix, 'bf')] = numpy.zeros((options['dim_token'] * 4,)).astype(config.floatX)

    params[_p(prefix, 'Wb')] = numpy.concatenate([ortho_weight(options['dim_token']),
                                                  ortho_weight(options['dim_token']),
                                                  ortho_weight(options['dim_token']),
                                                  ortho_weight(options['dim_token'])], axis=1)


    params[_p(prefix, 'Ub')] = numpy.concatenate([ortho_weight(options['dim_token']),
                                                  ortho_weight(options['dim_token']),
                                                  ortho_weight(options['dim_token']),
                                                  ortho_weight(options['dim_token'])], axis=1)

    params[_p(prefix, 'bb')] = numpy.zeros((options['dim_token'] * 4,)).astype(config.floatX)

    params[_p(prefix, 'V')] = rng.normal(scale=0.1, size=(options['dim_token'] * 1,)).astype(config.floatX)

    ################################################################################################
    

    ################################################################################################


    return params



def lstm_layer(tparams, inputs, masks, options, trng, prefix="lstm"):



    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step_f(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'Uf')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_token']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_token']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_token']))
        c = tensor.tanh(_slice(preact, 3, options['dim_token']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    def _step_f(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'Uf')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_token']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_token']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_token']))
        c = tensor.tanh(_slice(preact, 3, options['dim_token']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    def _step_b(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'Ub')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_token']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_token']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_token']))
        c = tensor.tanh(_slice(preact, 3, options['dim_token']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c


    state_0_f = tensor.dot(inputs[0], tparams[_p(prefix, 'Wf')]) + tparams[_p(prefix, 'bf')]
    state_1_f = tensor.dot(inputs[1], tparams[_p(prefix, 'Wf')]) + tparams[_p(prefix, 'bf')]
    state_2_f = tensor.dot(inputs[2], tparams[_p(prefix, 'Wf')]) + tparams[_p(prefix, 'bf')]

    state_0_b = tensor.dot(inputs[0], tparams[_p(prefix, 'Wb')]) + tparams[_p(prefix, 'bb')]
    state_1_b = tensor.dot(inputs[1], tparams[_p(prefix, 'Wb')]) + tparams[_p(prefix, 'bb')]
    state_2_b = tensor.dot(inputs[2], tparams[_p(prefix, 'Wb')]) + tparams[_p(prefix, 'bb')]

    dim_proj = options['dim_token']


    ##############################################################################################
    rval_f0, updates_f0 = theano.scan(_step_f,
                                sequences=[masks[0], state_0_f],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           inputs[0].shape[1],
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           inputs[0].shape[1],
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=inputs[0].shape[0])

    rval_b0, updates_b0 = theano.scan(_step_b,
                                sequences=[masks[0], state_0_b],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           inputs[0].shape[1],
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           inputs[0].shape[1],
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=inputs[0].shape[0],
                                go_backwards=True)

    proj_0 = rval_f0[0] + rval_b0[0][::-1]

    # Attention
    y_0 = (tensor.tanh(proj_0) * masks[0][:, :, None]) * tparams[_p(prefix, 'V')]
    y_0 = y_0.sum(axis=2).transpose()
    alpha = tensor.nnet.softmax(y_0).transpose()
    proj_0 = (proj_0 * masks[0][:, :, None]) * alpha[:, :, None]

    # Pooling
    proj_0 = (proj_0 * masks[0][:, :, None]).sum(axis=0)

    ##############################################################################################

    rval_f1, updates_f1 = theano.scan(_step_f,
                                sequences=[masks[1], state_1_f],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           inputs[1].shape[1],
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           inputs[1].shape[1],
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=inputs[1].shape[0])

    rval_b1, updates_b1 = theano.scan(_step_b,
                                sequences=[masks[1], state_1_b],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           inputs[1].shape[1],
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           inputs[1].shape[1],
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=inputs[1].shape[0],
                                go_backwards=True)

    proj_1 = rval_f1[0] + rval_b1[0][::-1]

    # Attention
    y_1 = (tensor.tanh(proj_1) * masks[1][:, :, None]) * tparams[_p(prefix, 'V')]
    y_1 = y_1.sum(axis=2).transpose()
    alpha_1 = tensor.nnet.softmax(y_1).transpose()
    proj_1 = (proj_1 * masks[1][:, :, None]) * alpha_1[:, :, None]

    # Pooling
    proj_1 = (proj_1 * masks[1][:, :, None]).sum(axis=0)
    
    ##############################################################################################
    rval_f2, updates_f2 = theano.scan(_step_f,
                                sequences=[masks[2], state_2_f],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           inputs[2].shape[1],
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           inputs[2].shape[1],
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=inputs[2].shape[0])

    rval_b2, updates_b2 = theano.scan(_step_b,
                                sequences=[masks[2], state_2_b],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           inputs[2].shape[1],
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           inputs[2].shape[1],
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=inputs[2].shape[0],
                                go_backwards=True)

    proj_2 = rval_f2[0] + rval_b2[0][::-1]

    # Attention
    y_2 = (tensor.tanh(proj_2) * masks[2][:, :, None]) * tparams[_p(prefix, 'V')]
    y_2 = y_2.sum(axis=2).transpose()
    alpha_2 = tensor.nnet.softmax(y_2).transpose()
    proj_2 = (proj_2 * masks[2][:, :, None]) * alpha_2[:, :, None]

    # Pooling
    proj_2 = (proj_2 * masks[2][:, :, None]).sum(axis=0)
    
    ##############################################################################################


    proj_0 = tensor.tanh(proj_0)
    proj_1 = tensor.tanh(proj_1)
    proj_2 = tensor.tanh(proj_2)
     
    return tensor.concatenate((proj_0, proj_1), axis=1), proj_2



def sgd(lr, tparams, grads, x, masks, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x[0], x[1], x[2], x[3], masks[0], masks[1], masks[2], masks[3], y[0], y[1]], 
                                    cost, updates=gsup, name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, masks, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x[0], x[1], x[2], x[3], masks[0], masks[1], masks[2], masks[3], y[0], y[1]], 
                                    cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, masks, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x[0], x[1], x[2], x[3], masks[0], masks[1], masks[2], masks[3], y[0], y[1]], 
                                    cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update




def momentum(lr, tparams, grads, x, masks, y, cost):
    
    rho = 0.9

    pre_step = [theano.shared(p.get_value() * 0., name='%s_pre_step' % k, 
                broadcastable = p.broadcastable)
               for k, p in tparams.items()]

    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x[0], x[1], x[2], x[3], masks[0], masks[1], masks[2], masks[3], y[0], y[1]], 
                                    cost, 
                                    updates=gsup,
                                    name='sgd_f_grad_shared')


    step = [rho * ps - lr * g for ps, g in zip(pre_step, gshared)]

    sup = [(ps, s) for ps, s in zip(pre_step, step)]

    pup = [(p, p + s) for p, s in zip(tparams.values(), step)]


    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup+sup,
                               name='sgd_f_update')

    return f_grad_shared, f_update



def nesterov_momentum(lr, tparams, grads, x, masks, y, cost):
    
    rho = 0.9

    pre_step = [theano.shared(p.get_value() * 0., name='%s_pre_step' % k, 
                broadcastable = p.broadcastable)
               for k, p in tparams.items()]

    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x[0], x[1], x[2], x[3], masks[0], masks[1], masks[2], masks[3], y[0], y[1]], 
                                    cost, 
                                    updates=gsup,
                                    name='sgd_f_grad_shared')


    step = [rho * ps - lr * g for ps, g in zip(pre_step, gshared)]

    sup = [(ps, s) for ps, s in zip(pre_step, step)]

    pup = [(p, p + rho * rho * s - (1+rho) * lr * g) for p, s, g in zip(tparams.values(), step, gshared)]


    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup+sup,
                               name='sgd_f_update')

    return f_grad_shared, f_update





def build_model(tparams, options):
    trng = RandomStreams(SEED)


    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x0 = tensor.matrix('x0', dtype='int32') #SIP Path
    mask0 = tensor.matrix('mask0', dtype=config.floatX)
    x1 = tensor.matrix('x1', dtype='int32') #RelSrc Path
    mask1 = tensor.matrix('mask1', dtype=config.floatX)
    x2 = tensor.matrix('x2', dtype='int32') #Cue Path
    mask2 = tensor.matrix('mask2', dtype=config.floatX)
    x3 = tensor.matrix('x3', dtype='int32') #child toks
    mask3 = tensor.matrix('mask3', dtype=config.floatX)
    y0 = tensor.vector('y0', dtype='int32')
    y1 = tensor.vector('y1', dtype='int32')
    

    sent_emb_0 = tparams['Wemb'][x0]
    sent_emb_1 = tparams['Wemb'][x1]
    sent_emb_2 = tparams['Wemb'][x2[2:, :]]
    sent_emb_3 = tparams['Wemb'][x3.transpose()]

    cueType_emb = tparams['CueTemb'][x2[0, :]]
    locdiff_emb = tparams['Lemb'][x2[1, :]]

    
    #LSTM Layer
    proj_0, proj_1 = lstm_layer(tparams,
                               inputs = [sent_emb_0, sent_emb_1, sent_emb_2],
                               masks  = [mask0, mask1, mask2[2:, :]],
                               options = options, trng = trng)

    proj_1 = tensor.concatenate((cueType_emb, locdiff_emb, proj_1), axis=1)

    #CNN Layer
    proj_2 = cnn_layer(tparams, sent_emb_3, mask3.transpose(), options=options, trng=trng)

    #
    proj_0 = tensor.concatenate((proj_0, proj_2), axis=1)


    #normalization
    #proj_0 = proj_0 / ((tensor.sqrt(tensor.sum(proj_0**2, axis=1)))[:,None] + 1e-8)
    #proj_1 = proj_1 / ((tensor.sqrt(tensor.sum(proj_1**2, axis=1)))[:,None] + 1e-8)

    #classifier
    proj_test_0 = proj_0
    proj_test_1 = proj_1


    proj_0 = proj_0 * dropout_mask_1D(proj_0, 1, 0.8, trng)
    proj_1 = proj_1 * dropout_mask_1D(proj_1, 1, 0.8, trng)
    

    pred0 = tensor.nnet.softmax(tensor.dot(proj_0, tparams['W30']) + tparams['b30'])
    pred1 = tensor.nnet.softmax(tensor.dot(proj_1, tparams['W31']) + tparams['b31'])
    pred_test0 = tensor.nnet.softmax(tensor.dot(proj_test_0, tparams['W30']) + tparams['b30'])
    pred_test1 = tensor.nnet.softmax(tensor.dot(proj_test_1, tparams['W31']) + tparams['b31'])

    '''pred0 = tensor.clip(pred0, 0.001, 0.999)
    pred1 = tensor.clip(pred1, 0.001, 0.999)
    pred_test0 = tensor.clip(pred_test0, 0.001, 0.999)
    pred_test1 = tensor.clip(pred_test1, 0.001, 0.999)'''

    f_pred_prob = theano.function(inputs=[x0, x1, x2, x3, mask0, mask1, mask2, mask3], 
                                outputs=[pred0.max(axis=1), pred1.max(axis=1)], 
                                name='f_pred_prob')
    f_pred = theano.function(inputs=[x0, x1, x2, x3, mask0, mask1, mask2, mask3],
                             outputs=[pred0.argmax(axis=1), pred1.argmax(axis=1)], 
                             name='f_pred')

    f_pred_prob_test = theano.function(inputs=[x0, x1, x2, x3, mask0, mask1, mask2, mask3],
                                       outputs=[pred_test0.max(axis=1), pred_test1.max(axis=1)], 
                                       name='f_pred_prob_test')
    f_pred_test = theano.function(inputs=[x0, x1, x2, x3, mask0, mask1, mask2, mask3],
                                  outputs=[pred_test0.argmax(axis=1), pred_test1.argmax(axis=1)], 
                                  name='f_pred_test')


    off = 1e-8

    lamda_value = theano.shared(1e-4)
    theta_sum = theano.shared(0.)
    for kk, pp in tparams.items():
        theta_sum += tensor.sum(tparams[kk]**2)

    _e = 0.75
    cost = - _e *       tensor.mean(tensor.log(pred0[tensor.arange(y0.shape[0]), y0] + off)) + \
           - (1 - _e) * tensor.mean(tensor.log(pred1[tensor.arange(y1.shape[0]), y1] + off)) + \
                        0.5 * lamda_value * theta_sum


    return use_noise, [x0,x1,x2,x3], [mask0,mask1,mask2,mask3], [y0,y1], \
                f_pred_prob, f_pred, cost, f_pred_prob_test, f_pred_test


#def pred_probs(f_pred_prob, prepare_data, data, iterator, options, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    '''n_samples = len(data[0])
    probs = numpy.zeros((n_samples, options['ydim'])).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y, x_maxlen = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs'''


#def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    '''valid_err = 0
    for _, test_index in iterator:
        x0, mask0 = prepare_data([data[0][t] for t in test_index])
        x1, mask1 = prepare_data([data[1][t] for t in test_index])
        x2, mask2 = prepare_data([data[2][t] for t in test_index])
        
        pred_labels = f_pred([x0, x1, x2], [mask0, mask1, mask2])
        
        targets = [numpy.array(data[3])[test_index], numpy.array(data[4])[test_index]]
        
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err'''


def output_pred_labels(f_pred, f_pred_prob, prepare_data, data, iterator, verbose, path):
    f = open(path,'w')
    for _, test_index in iterator:
        x0, mask0 = prepare_data([data[0][t] for t in test_index])
        x1, mask1 = prepare_data([data[1][t] for t in test_index])
        x2, mask2 = prepare_data([data[2][t] for t in test_index])
        x3, mask3 = prepare_data([data[3][t] for t in test_index])
        
        pred_labels = f_pred(x0, x1, x2, x3, mask0, mask1, mask2, mask3)

        pred_maxProbs = f_pred_prob(x0, x1, x2, x3, mask0, mask1, mask2, mask3)

        #pred_maxProbSum = pred_maxProbs[0] + pred_maxProbs[1]

        for i in range(pred_labels[0].shape[0]):
            f.write(str(pred_labels[0][i])+' '+str(pred_labels[1][i])+' '+
                    str(pred_maxProbs[0][i])+' '+str(pred_maxProbs[1][i])+'\n')

    f.close();


def train_cnn(

    # Hyper-Parameters

    dim_token = 100,  # word embeding dimension
    dim_locDiff = 10, # location difference dimension
    dim_cueType = 10, #

    cnn_n1 = 50,
    n2 = 10 + 10 + 100,
    ydim0 = 3,
    ydim1 = 3,
    #win_size = 3,

    #maxTokens1 = 60, # maximum tokens in sentence 1

    n_cueTypes = 5,
    n_words = 4000,  # Vocabulary size
    n_locDiffs = 108,  # Location difference size

    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=300,  # The maximum number of epoch to run
    #dispFreq=10,  # Display to stdout the training progress every N updates
    #decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.01,  # Learning rate for sgd (not used for adadelta and rmsprop)
    
    optimizer = momentum,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).


    #maxlen=1000,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.


    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    #reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1
):

    # Model options
    model_options = locals().copy()
    print('----------------------------------------------')
    print("model options", model_options)
    print('----------------------------------------------')

    #load_data, prepare_data = get_dataset(dataset)

    print('Loading data ... ... ...')
    train, valid, test = data.load_data(path='../mydata.pkl.gz',
                                n_words=n_words, valid_portion=0.)
    '''if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])'''

    

    print('Building model ... ... ...')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options, Wemb_value=data.read_gz_file("../matrix.pkl.gz"))

    '''if reload_model:
        load_params('cnn_model.npz', params)'''

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, masks, y,
     f_pred_prob, f_pred,
     cost, 
     f_pred_prob_test, f_pred_test) = build_model(tparams, model_options)

    '''if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay'''

    f_cost = theano.function([x[0], x[1], x[2], x[3], masks[0], masks[1], masks[2], masks[3], y[0], y[1]], 
                            cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x[0], x[1], x[2], x[3], masks[0], masks[1], masks[2], masks[3], y[0], y[1]], 
                            grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, masks, y, cost)

    #print('Optimization')
    print('training ... ... ...')

    kf_valid = get_minibatches_idx(len(valid[0]), batch_size)
    kf_test  = get_minibatches_idx(len(test[0]), batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    #history_errs = []
    best_p = None
    bad_counter = 0

    '''if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size'''

    last_training_sum_costs = numpy.inf

    uidx = 0  # the number of update done
    estop = False  # early stop
    #start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            training_sum_costs = 0

            for train_batch_idx, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                x_0 = [train[0][t] for t in train_index]
                x_1 = [train[1][t] for t in train_index]
                x_2 = [train[2][t] for t in train_index]
                x_3 = [train[3][t] for t in train_index]
                y_0 = [train[4][t] for t in train_index]
                y_1 = [train[5][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # 
                # Return something of shape (minibatch maxlen, n samples)
                x_0, mask_0 = data.prepare_data(x_0)
                x_1, mask_1 = data.prepare_data(x_1)
                x_2, mask_2 = data.prepare_data(x_2)
                x_3, mask_3 = data.prepare_data(x_3)
                y_0 = numpy.asarray(y_0, dtype='int32')
                y_1 = numpy.asarray(y_1, dtype='int32')
                
                n_samples += x_0.shape[1]

                if train_batch_idx % 100 == 0 or train_batch_idx == len(kf) - 1:
                    print("%d/%d training bacthes @ epoch = %d" % (train_batch_idx, len(kf), eidx))

                cost = f_grad_shared(x_0, x_1, x_2, x_3, mask_0, mask_1, mask_2, mask_3, y_0, y_1)
                f_update(lrate)

                training_sum_costs += cost

            print("sum of costs of all the training samples = ",training_sum_costs,"@ epoch = ", eidx)

            if numpy.isnan(training_sum_costs) or numpy.isinf(training_sum_costs):
                print('bad cost detected: ', training_sum_costs)
                print('End of Program')
                break


            print('outputing predicted labels of test set ... ... ...')

            output_pred_labels(f_pred_test, f_pred_prob_test,
                data.prepare_data, test, kf_test, 
                verbose=False, path="test_pred_labels.txt")

            if training_sum_costs >= last_training_sum_costs * 0.99:
                bad_counter += 1
                if bad_counter == patience / 2:
                    lrate /= 4.

            last_training_sum_costs = training_sum_costs


            print('bad counter for early stopping : %d/%d' % (bad_counter, patience))
            print('learning rate = ', lrate)
            print('--------------------------------------------------')


            if bad_counter >= patience:
                print('Early Stop!')
                estop = True
                break

            if estop:
                break


    except KeyboardInterrupt:
        print("Training interupted")



if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_cnn(
        max_epochs=300,
        test_size=-1,
    )
