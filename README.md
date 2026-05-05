# Multimodal_Models_CrypticBio

A repository for the development and benchmarking of different multimodal approaches for categorizing visually confusing species.

## SINR setup

Required for `--e sinr` experiments:

```bash
#at root directory
mkdir -p external && cd external
git clone https://github.com/elijahcole/sinr.git
cd sinr && mkdir -p pretrained_models
# Download model_an_full_input_enc_sin_cos_distilled_from_env.pt
# from the SINR repo's web_app/README.md into pretrained_models/
```