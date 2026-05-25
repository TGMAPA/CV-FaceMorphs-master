# Execute pair csv generation
python -m ControlledMorphGeneration.t_sne_sample_set_extraction > ControlledMorphGeneration/t_sne_sample_set_extraction.out

# Execute morph generation with csv
python -m ControlledMorphGeneration.ControlledMorphGeneration > ControlledMorphGeneration/ControlledMorphGeneration.out