from .base import BaseSingleSpeciesDisease
from .transition import (
                         TransitionCustomProbPermanentImmunity)
from .transmission import TransmissionByContact
from ...utils.decorators import sampy_class


@sampy_class
class ContactCustomProbTransitionPermanentImmunity(BaseSingleSpeciesDisease,
                                                   TransitionCustomProbPermanentImmunity,
                                                   TransmissionByContact):
    """
    Basic disease, transmission by direct contact (sharing a same vertex), transition between disease states
    encoded by user given arrays of probabilities, and permanent immunity.
    """
    def __init__(self, **kwargs):
        pass


# class ContactRandomUniformTransition(BaseSingleSpeciesDisease,
#                                      TransitionWithUniformProb,
#                                      TransmissionByContact):
#     def __init__(self, disease_name, host):
#         super().__init__(disease_name=disease_name, host=host)
#
#
# class ContactDeterministicTransitionPermanentImmunity(BaseSingleSpeciesDisease,
#                                                       TransitionDeterministicCounterWithPermanentImmunity,
#                                                       TransmissionByContact):
#     def __init__(self, disease_name, host):
#         super().__init__(disease_name=disease_name, host=host)
