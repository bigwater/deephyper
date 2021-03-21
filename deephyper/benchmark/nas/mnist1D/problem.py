from deephyper.problem import NaProblem
from deephyper.benchmark.nas.mnist1D.load_data import load_data
from deepspace.tabular import OneLayerFactory
from deephyper.nas.space.auto_keras_search_space import AutoKSearchSpace
from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op.op1d import (Activation, Conv1D,
                                                      Dense, Dropout, Flatten,
                                                      Identity, MaxPooling1D)


def add_conv_op_(node):
    # node.add_op(Identity())
    node.add_op(Conv1D(filter_size=3, num_filters=8))
    node.add_op(Conv1D(filter_size=4, num_filters=8))
    node.add_op(Conv1D(filter_size=5, num_filters=8))
    node.add_op(Conv1D(filter_size=6, num_filters=8))

def add_dense_op_(node):
    # node.add_op(Identity())
    node.add_op(Dense(units=10))
    node.add_op(Dense(units=50))
    node.add_op(Dense(units=100))
    node.add_op(Dense(units=200))
    node.add_op(Dense(units=250))
    node.add_op(Dense(units=500))
    node.add_op(Dense(units=750))
    node.add_op(Dense(units=1000))

def add_activation_op_(node):
    # node.add_op(Identity())
    node.add_op(Activation(activation='relu'))
    node.add_op(Activation(activation='tanh'))
    node.add_op(Activation(activation='sigmoid'))



def create_search_space(input_shape=(10,), output_shape=(1,), **kwargs):
    struct = AutoKSearchSpace(input_shape, output_shape, regression=False)

    # n1 = VariableNode('N1ConvOp')
    # add_conv_op_(n1)
    # struct.connect(struct.input_nodes[0], n1)

    n00 = VariableNode('N8DenseOp')
    add_dense_op_(n00)
    struct.connect(struct.input_nodes[0], n00)

    n90 = VariableNode('N9ActivOp')
    add_activation_op_(n90)
    struct.connect(n00, n90)


    n8 = VariableNode('N8DenseOp')
    add_dense_op_(n8)
    struct.connect(n90, n8)

    n9 = VariableNode('N9ActivOp')
    add_activation_op_(n9)
    struct.connect(n8, n9)

    return struct

    # return OneLayerFactory()(input_shape, output_shape, **kwargs)


Problem = NaProblem()

Problem.load_data(load_data)

Problem.search_space(create_search_space)

Problem.hyperparameters(
    batch_size=32, learning_rate=0.01, optimizer="adam", num_epochs=10
)

Problem.loss("categorical_crossentropy")

Problem.metrics(["acc"])

Problem.objective("val_acc")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)
