# Introduction

This repository provides ensemble machine learning models for predicting nephrotoxicity of compounds, including natural products from traditional Chinese medicine (TCM). These models utilize structural fingerprints and molecular descriptors of the compounds to make accurate predictions. Overall, the repository enables efficient and accurate nephrotoxicity prediction, providing valuable insights for researchers working with natural compounds or traditional Chinese medicine.

![image-20241130123402793](https://raw.githubusercontent.com/lishensuo/images2/main/img01/image-20241130123402793.png)



---

# Preparation

## Python Environment

We recommend using **conda** to create the Python environment based on the `requirement.txt` file. 

The main dependencies include:

- **padelpy**, **mordred**, and **rdkit**: For chemical feature representation.
- **autogluon**: For machine learning modeling.

## Data

The repository includes the following necessary data:

- **`fingerprints_xml/`**: A folder containing the required files to encode 12 PaDEL fingerprint features, which can be downloaded via the [link](https://github.com/dataprofessor/padel).
- **`fingerprints_metainfo.csv`**: A file with brief annotations for all the chemical features used to build the models.
- **`model100_metainfo.csv`**: A file summarizing the modeling results for 100 datasets with 5-fold cross-validation.

### Model Files

Due to GitHub's storage limitations, the model files are hosted on Google Drive. Users can download the compressed file (`.gz`) from the provided [link](https://drive.google.com/drive/folders/1t8AP_wJZ5lTmlJbj69a9kInBRpqogIOB?usp=drive_link) and decompress it locally.

---

# Tutorial

Once the environment and data are set up, predicting the nephrotoxicity of a compound is straightforward, especially for natural ingredients in TCM. The only required inputs are the **PubChem ID** and its **SMILES** string.

The script will automatically:

1. Encode the compound's features.
2. Make predictions.

### Example Workflow

1. Provide the **PubChem ID** and **SMILES** of the compound.
2. Run the script.
3. The predictive nephrotoxicity score will be displayed, and the results will be saved locally.

---




```python
python nephrotoxicity_model.py \
    --cid 6681 \
    --smile "C1=CC=C2C(=C1)C(=O)C3=C(C2=O)C(=C(C=C3Br)Br)N" \
	--p_work "./task" \
    --p_model "./path/to/nephro_toxi_model" \
    --p_model_meta "./model100_metainfo.csv" \
    --p_xml "./fingerprints_xml" \
    --p_fpmeta "./fingerprints_metainfo.csv"
```

- Log record:

```
******************** Step 1/2: Prepare fingerprint feature for the input compound. ********************
The compound CID    : 6681
The compound SMILES : C1=CC=C2C(=C1)C(=O)C3=C(C2=O)C(=C(C=C3Br)Br)N
==> Start to encode AtomPairs2DCount [1/12].
==> Start to encode AtomPairs2D [2/12].
==> Start to encode EState [3/12].
==> Start to encode CDKextended [4/12].
==> Start to encode CDK [5/12].
==> Start to encode CDKgraphonly [6/12].
==> Start to encode KlekotaRothCount [7/12].
==> Start to encode KlekotaRoth [8/12].
==> Start to encode MACCS [9/12].
==> Start to encode PubChem [10/12].
==> Start to encode SubstructureCount [11/12].
==> Start to encode Substructure [12/12].
Padel fingerprints encode has been done and stored in task/cid_fp/6681
The compound CID    : 6681
The compound SMILES : C1=CC=C2C(=C1)C(=O)C3=C(C2=O)C(=C(C=C3Br)Br)N
==> Start to encode PaDEL Descriptor.
==> Start to encode Mordred Descriptor.
Padel & Mordred descriptors encode have been done and stored in task/cid_desc/6681
******************** Step 2/2: Predict the nephrotoxicity of the input compund. ********************
===> Start to predict the nephrotoxicity via set_0 model (SubstructureCount_fold0)
===> Start to predict the nephrotoxicity via set_0 model (mordred_fold1)
===> Start to predict the nephrotoxicity via set_0 model (AtomPairs2D_fold2)
===> Start to predict the nephrotoxicity via set_0 model (mordred_fold3)
===> Start to predict the nephrotoxicity via set_0 model (padel_fold4)
===> Start to predict the nephrotoxicity via set_1 model (PubChem_fold0)
===> Start to predict the nephrotoxicity via set_1 model (padel_fold1)
===> Start to predict the nephrotoxicity via set_1 model (AtomPairs2DCount_fold2)
===> Start to predict the nephrotoxicity via set_1 model (padel_fold3)
===> Start to predict the nephrotoxicity via set_1 model (AtomPairs2DCount_fold4)
######## omit some redundancy log ########
===> Start to predict the nephrotoxicity via set_99 model (SubstructureCount_fold0)
===> Start to predict the nephrotoxicity via set_99 model (PubChem_fold1)
===> Start to predict the nephrotoxicity via set_99 model (mordred_fold2)
===> Start to predict the nephrotoxicity via set_99 model (KlekotaRothCount_fold3)
===> Start to predict the nephrotoxicity via set_99 model (KlekotaRoth_fold4)
******************** Done! The overall estimated nephrotoxicity probality of the input compund is 0.639. ********************

```

- Saved results

```python
ls ./task/cid_pred/6681
# overall_score.csv  raw_score.csv
```
