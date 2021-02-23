from .vaccination import (BaseVaccinationSingleSpeciesDisease,
                          VaccinationSingleSpeciesDiseaseFixedDuration)
from ..utils.decorators import sampy_class


@sampy_class
class BasicVaccination(BaseVaccinationSingleSpeciesDisease,
                       VaccinationSingleSpeciesDiseaseFixedDuration):
    def __init__(self, **kwargs):
        pass
