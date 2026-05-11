# Multimodal_Models_CrypticBio

A repository for the development and bechmarking of different multimodal aproaches for categorizing visually confusing species.

The CrypticBio Common dataset was accessed through the Hugging Face Datasets library, using version ``4.8.5``. The identifier *gmanolache/CrypticBio* was used, loading the file *CrypticBio-Benchmark/CrypticBio-Common.csv* as the train split.

Python version: ``3.11.15``

To run the experiments, two branches are required.

To run the taxonomy and modality experiments, follow these steps:

* Clone/download the repository
* Create an environment
* Install the dependencies from *requirements.txt*, using ``pip install -r requirements.txt``
* To run the taxonomy and modality experiments, use: ``python taxonomy_experiment.py --e ["location", "date", "both"] --level ["kingdom", "phylum", "class", "order", "family", "genus", "scientificName"]``
* This experiment will train the MLP network on date or location data, and evaluate the test set with both BioCLIP-2 and the fused model.

To run the SINR experiment, follow these steps:

* Follow the first three steps from previous instructions
* Switch to the SINR branch: ``git checkout Extra_experimetn``
* To create the needed file structure for SINR:
* Create the needed folders: ``mkdir -p external/sinr``
* Clone the SINR GitHub: ``git clone https://github.com/elijahcole/sinr.git external/sinr``
* Download the pre-trained model file: ``wget "https://data.caltech.edu/records/dk5g7-rhq64/files/pretrained_models.zip?download=1" -O pretrained_models.zip``
* Unzip the file: ``unzip pretrained_models.zip -d external/sinr/``
* Then, run the experiment using: ``python sinr_location_experiment.py``

All environmental details are found in the ``requirements.txt`` file.

To run statistics on the common dataset, follow these steps:

* Follow the first three steps from the first instructions
* Move into the folder ``‎statistics``
* Run the Jupyter notebook file as  ``jupyter nbconvert --to notebook --inplace --execute statistics.ipynb``
* 2 output files will be created in a new folder ``output``, all other plots are visable in the Jupyter notebook file ``statistics.ipynb``. 
