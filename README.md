# Multimodal_Models_CrypticBio

A repository for the development and bechmarking of different multimodal aproaches for categorizing visually confusing species.

The CrypticBio Common dataset was accessed through the Hugging Face Datasets library, using version 4.8.5. The identifier gmanolache/CrypticBio was used, loading the file CrypticBio-Benchmark/CrypticBio-Common.csv as the train split.

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


Include:

\begin{itemize}
    \item Link to code repository or submitted code package
    \item Exact dataset versions and access links
    \item Environment details (Python version, library versions)
    \item Execution steps
    \item Random seeds
    \item Any known issues or limitations affecting reproducibility
\end{itemize}

A possible structure:
\begin{enumerate}
    \item Clone/download the repository.
    \item Install dependencies from \texttt{requirements.txt}.
    \item Download the data from the provided links.
    \item Run preprocessing scripts in the following order: [list scripts].
    \item Run training with: \texttt{python train.py --config config.yaml}
    \item Run evaluation with: \texttt{python eval.py --checkpoint ...}
\end{enumerate}
