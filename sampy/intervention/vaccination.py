import numpy as np
from ..pandas_xs.pandas_xs import DataFrameXS


class BaseVaccinationSingleSpeciesDisease:
    def __init__(self, disease=None, **kwargs):
        if disease is None:
            raise ValueError(
                "No disease object provided for the vaccination. Should be provided using kwarg 'disease'.")
        self.disease = disease
        self.target_species = self.disease.host
        self.target_species.df_population['vaccinated'] = False


class VaccinationSingleSpeciesDiseaseFixedDurationIgnoreNaturallyImmune:
    def __init__(self, duration_vaccine=None, **kwargs):
        if duration_vaccine is None:
            raise ValueError(
                "No duration provided for the vaccination. Should be provided using kwarg 'duration_vaccine'.")
        self.duration_vaccine = int(duration_vaccine)
