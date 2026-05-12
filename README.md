# Multimodal_Models_CrypticBio - SINR Branch

A repository for the development and benchmarking of different multimodal approaches for categorizing visually confusing species.
This brnch is a modified version of mian that allows running the SINR epxeriemtns as well as the MPL experiments on different CrypticBio benchmark sets.

## SINR setup

Required for `--e sinr` experiments

After isntalling requirements.txt:

```bash
#at root directory
mkdir -p external && cd external
git clone https://github.com/elijahcole/sinr.git
cd sinr && mkdir -p pretrained_models
# Download model_an_full_input_enc_sin_cos_distilled_from_env.pt
#Linux
wget "https://data.caltech.edu/records/dk5g7-rhq64/files/pretrained_models.zip?download=1" -O pretrained_models.zip
unzip -j pretrained_models.zip "pretrained_models/model_an_full_input_enc_sin_cos_distilled_from_env.pt" -d pretrained_models/
rm pretrained_models.zip
#Or windows(tested)
curl -L -o pretrained_models.zip "https://data.caltech.edu/records/dk5g7-rhq64/files/pretrained_models.zip?download=1"
tar -xf pretrained_models.zip pretrained_models/model_an_full_input_enc_sin_cos_distilled_from_env.pt
del pretrained_models.zip
```
Navigate back to root. And now you are able to run all experimets!
Example usage: 
```bash
python main.py --e sinr --benchmark common
```
