from sampy.utils.decorators import use_debug_mode
from sampy.pandas_xs.pandas_xs import DataFrameXS

from sampy.graph.builtin_graph import SquareGridWithDiag
from sampy.agent.builtin_agent import BasicMammal
from sampy.disease.single_species.builtin_disease import ContactCustomProbTransitionPermanentImmunity

import numpy as np


use_debug_mode(DataFrameXS)
use_debug_mode(SquareGridWithDiag)
use_debug_mode(BasicMammal)

# graph
my_graph = SquareGridWithDiag(shape=(100, 100))
my_graph.create_vertex_attribute('K', 10.)

# agent
agents = BasicMammal(graph=my_graph)
print(agents.df_population.list_col_name)

# # disease
# disease = ContactCustomProbTransitionPermanentImmunity(disease_name='rabies', host=agents)
# print(agents.df_population.list_col_name)


