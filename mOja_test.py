import nengo
import numpy as np
from extras import *
from learning_rules import SimmOja, mOja
from nengo.learning_rules import Oja
from neurons import AdaptiveLIFLateralInhibition
import re

seed = 4
np.random.seed( seed )
beta = 1
neurons = 4
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
    # pos_memr_probe = nengo.Probe( conn.learning_rule, "pos_memristors" )
    # neg_memr_probe = nengo.Probe( conn.learning_rule, "neg_memristors" )

# with nengo.Simulator( model, seed=seed ) as sim:
#     # training
#     sim.run( time / 2 )
#
#     modify_learning_rate( sim, conn, SimmOja, new_lr=0 )
#     # testing
#     sim.run( time / 2 )
#
# probe_train = sim.data[ ens_probe ][ :int( time / sim.dt / 2 ) ]
# probe_test = sim.data[ ens_probe ][ int( time / sim.dt / 2 ): ]

with nengo.Simulator( model, seed=seed ) as sim_train:
    # training
    sim_train.run( time / 2 )

conn.transform = sim_train.data[ weight_probe ][ -1 ].squeeze()
conn.learning_rule_type = mOja( gain=1e6, beta=beta, noisy=0, learning_rate=0,
                                initial_state={ "weights": sim_train.data[ weight_probe ][ -1 ].squeeze() }
                                )
# with model:
#     # remove probes from model
#     # model.objects[ nengo.Probe ] = [ x for x in model.objects[ nengo.Probe ] if not re.match( r"\w+memristors",
#     # x.attr ) ]
#     probes_to_remove = [ x for x in model.probes if re.match( r"\w+memristors", x.attr ) ]
#     for probe in probes_to_remove:
#         model.probes.remove( probe )
#     model.probes.append( nengo.Probe( conn.learning_rule, "pos_memristors" ) )
#     model.probes.append( nengo.Probe( conn.learning_rule, "neg_memristors" ) )
# conn.learning_rule_type = Oja( beta=beta, learning_rate=0 )
# modify_learning_rate( sim, conn, SimmOja, new_lr=0 )
# rule_type = conn.learning_rule_type
# type( rule_type ).learning_rate.data[ rule_type ] = 0

with nengo.Simulator( model, seed=seed ) as sim_test:
    # with nengo.Simulator( model, seed=seed ) as sim_test:
    # testing
    sim_test.run( time / 2 )

combined_probes = combine_probes( [ sim_train, sim_test ] )
combined_trange = combine_tranges( [ sim_train, sim_test ] )

fig1 = neural_activity_plot( combined_probes[ ens_probe ], combined_trange )
fig1.show()
plt.set_cmap( "jet" )
fig2, ax = plt.subplots( 1 )
ax.matshow( combined_probes[ weight_probe ][ -1 ] )
fig2.show()
print()
print( combined_probes[ weight_probe ][ -1 ] )

plotter = Plotter( combined_trange, ens.n_neurons, ens.n_neurons, 1,
                   time / 2,
                   0.001,
                   plot_size=(13, 7),
                   dpi=300,
                   pre_alpha=0.3
                   )
# plotter.plot_weights_over_time( sim.data[ pos_memr_probe ], sim.data[ neg_memr_probe ], plot_all=True ).show()
# plotter.plot_values_over_time( sim.data[ pos_memr_probe ], sim.data[ neg_memr_probe ],
#                                value="conductance", plot_all=True ).show()

print( "Classification accuracy:" )
avg_act = np.mean( combined_probes[ ens_probe ][ int( time / sim_test.dt / 2 ): ], axis=0 )
count = neurons
for i, (neur, act) in enumerate( zip( train_neurons, avg_act ) ):
    if neur == -2 and act > 100 or neur == 2 and act < 100:
        count -= 1
        print( f"Neuron {i}: {neur}->{act}" )
print( f"Accuracy: {(count / neurons) * 100} %" )
