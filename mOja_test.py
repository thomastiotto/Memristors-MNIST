import nengo
import numpy as np
from extras import *
from learning_rules import mOja, mPES
from neurons import AdaptiveLIFLateralInhibition

seed = None
np.random.seed( seed )
beta = 1
neurons = 20
time = 10

train_neurons = np.random.randint( 2, size=neurons )
train_neurons[ train_neurons == 0 ] = -1
train_neurons *= 2

model = nengo.Network( seed=seed )
with model:
    inp = nengo.Node( lambda t, x: train_neurons if t < time / 2 else [ -2 ] * neurons, size_in=1 )
    ens = nengo.Ensemble( neurons, 1,
                          # neuron_type=AdaptiveLIFLateralInhibition(),
                          seed=seed )
    nengo.Connection( inp, ens.neurons, seed=seed )
    conn = nengo.Connection( ens.neurons, ens.neurons,
                             # learning_rule_type=nengo.learning_rules.Oja( beta=beta ),
                             learning_rule_type=mOja( gain=1e6, beta=beta, noisy=0 ),
                             # transform=np.random.random( (pre.n_neurons, pre.n_neurons) ),
                             transform=np.zeros( (ens.n_neurons, ens.n_neurons) ),
                             # transform=np.eye( pre.n_neurons ),
                             seed=seed
                             )
    # conn_inh = nengo.Connection( pre.neurons, pre.neurons,
    #                              transform=-2 * (np.ones( (pre.n_neurons, pre.n_neurons) ) - np.eye( pre.n_neurons )),
    #                              seed=seed
    #                              )

    ens_probe = nengo.Probe( ens.neurons )
    weight_probe = nengo.Probe( conn, "weights" )

with nengo.Simulator( model, seed=seed ) as sim:
    sim.run( time )

fig1 = neural_activity_plot( sim.data[ ens_probe ], sim.trange() )
fig1.show()
plt.set_cmap( "jet" )
fig2, ax = plt.subplots( 1 )
ax.matshow( sim.data[ weight_probe ][ -1 ] )
fig2.show()
print()
print( sim.data[ weight_probe ][ -1 ] )

avg_act = np.mean( sim.data[ ens_probe ][ int( time / sim.dt / 2 ): ], axis=0 )
count = neurons
for i, (neur, act) in enumerate( zip( train_neurons, avg_act ) ):
    if neur == -2 and act > 100 or neur == 2 and act < 100:
        count -= 1
        print( f"Neuron {i}: {neur}->{act}" )
print( f"Accuracy: {(count / neurons) * 100} %" )
