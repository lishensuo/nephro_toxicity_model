import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from tqdm import tqdm
import argparse
from padelpy import padeldescriptor, from_smiles
from autogluon.tabular import TabularDataset, TabularPredictor
from rdkit import Chem
from mordred import Calculator, descriptors
calc = Calculator(descriptors, ignore_3D=True)


# https://github.com/dataprofessor/padel/blob/main/fingerprints_xml.zip

# p_main = Path("data/middle/fp_padel")
# p_xml = Path("/home/lishensuo/PROJECT/Toxicity/data/feature/fingerp/fingerprints_xml")
# fp_meta = "/home/lishensuo/PROJECT/Toxicity/data/feature/merge/All_feat_meta19793.csv"


def get_padel_fp(cid, smile, p_main, p_xml, p_fpmeta):

    cid = str(cid) if isinstance(cid, int) else cid  

    print("The compound CID    : " + cid)
    print("The compound SMILES : " + smile)

    tmp_smi = Path(tempfile.gettempdir()) / f"{cid}.smi"

    FP_list = ['AtomPairs2DCount', # 781 +
         'AtomPairs2D', # 781
         'EState', #80 *
         'CDKextended', #1025 *
         'CDK', #1025 *
         'CDKgraphonly', #1025 *
         'KlekotaRothCount', #4861 * +
         'KlekotaRoth', #4861 * +
         'MACCS', #167
         'PubChem', #882
         'SubstructureCount', #308 +
         'Substructure'] #308

    p_main_cid = Path("./deploy/demo") / 'cid_fp' / cid
    if not p_main_cid.exists() or not any(p_main_cid.iterdir()):
    # if len(os.listdir(dir_fp)) != 12:
        p_main_cid.mkdir(parents=True, exist_ok=True)

        with open(tmp_smi, "w") as f:
            f.write(smile + "\t" + cid)

        xml_files = [xml for xml in p_xml.iterdir()]
        xml_files.sort()
        fp_xml_dict = dict(zip(FP_list, xml_files))
        fp_meta = pd.read_csv(p_fpmeta)

        for i, FP in enumerate(FP_list) :
            print(f"==> Start to encode {FP} [{i+1}/{len(FP_list)}].")
            # FP = FP_list[0]
            output_file = p_main_cid /  f'{FP}.csv'
            if output_file.exists(): 
                continue
            try:
                padeldescriptor(mol_dir=tmp_smi, 
                                d_file=output_file, 
                                descriptortypes= fp_xml_dict[FP],
                                detectaromaticity=True,
                                standardizenitro=True,
                                standardizetautomers=True,
                                removesalt=True,
                                log=False,
                                fingerprints=True)
            except:
                print("==> Failed to code, return None")
                fp_meta_sub = fp_meta[fp_meta.Type.isin([FP])] #查询特定指纹的长度
                output_df = pd.DataFrame([cid] + [None]*fp_meta_sub.shape[0]).transpose() 
                output_df.columns = ["Name"] + list(fp_meta_sub.Name)
                output_df.to_csv(output_file, index=False)
        if tmp_smi.exists():
            os.remove(tmp_smi)


    print(f"Padel fingerprints encode has been done and stored in {p_main_cid}" )


def get_padel_desc(cid, smile, p_main):

    cid = str(cid) if isinstance(cid, int) else cid  

    print("The compound CID    : " + cid)
    print("The compound SMILES : " + smile)

    p_main_cid = p_main / 'cid_desc' / cid

    if not p_main_cid.exists() or not any(p_main_cid.iterdir()):
        p_main_cid.mkdir(parents=True, exist_ok=True)
        ## Padel type
        print(f"==> Start to encode PaDEL Descriptor.")
        try:
            descriptors = from_smiles(smile)
        except RuntimeError:
            print("time out, return None")
            example_desc = from_smiles("ccc")
            descriptors = dict(zip(example_desc.keys(),[None]*len(example_desc.keys())))
        output_df = pd.DataFrame(descriptors, index = [cid])
        output_df.to_csv(p_main_cid / 'PaDEL_Descriptor.csv')

        ## Mordred type
        print(f"==> Start to encode Mordred Descriptor.")
        mol = Chem.MolFromSmiles(smile)
        
        try:
            output_df2 = pd.DataFrame(calc(mol)).transpose()
        except TypeError:
            output_df2 = pd.DataFrame([[None] * len(calc.descriptors)])
        output_df2.index = [cid]
        output_df2.to_csv(p_main_cid /  'Mordred_Descriptor.csv')

    print(f"Padel & Mordred descriptors encode have been done and stored in {p_main_cid}" )


def predict_nephrotoxicity(cid, smile, p_main, p_model, p_model_meta):
    cid = str(cid) if isinstance(cid, int) else cid  
    model_meta = pd.read_csv(p_model_meta)

    pred_list = []
    for i in tqdm(range(model_meta.shape[0])):
        set_id = model_meta["set"][i]
        model_type = model_meta["model"][i]
        feat_type = model_meta["feat_type"][i]
        fold = model_meta["fold"][i]
        
        p_model_choose = p_model / set_id / f"{feat_type}_fold{str(fold)}"
        print(f"===> Start to predict the nephrotoxicity via {set_id} model ({p_model_choose.name})")

        model = TabularPredictor.load(p_model_choose)

        feat_fls = [fl for fl in (p_main / "cid_fp" / cid).iterdir()] + \
                    [fl for fl in (p_main / "cid_desc" / cid).iterdir()]
        feat_types = ['AtomPairs2DCount', 'AtomPairs2D', 'EState', 'CDKextended', 'CDK',
                       'CDKgraphonly', 'KlekotaRothCount', 'KlekotaRoth', 'MACCS', 'PubChem', 
                       'SubstructureCount', 'Substructure'] + ['padel', 'mordred']
        feat_dict = {x : y for x, y in zip(feat_types, feat_fls)}

        feat_data = pd.read_csv(feat_dict[feat_type])
        if feat_type=="padel":
            feat_data.columns = [col.replace('-', '.') for col in feat_data.columns]
        elif feat_type=="mordred":
            feat_data.columns = ["X" + str(col) for col in feat_data.columns]

        feat_data_tab = TabularDataset(feat_data)
        pred = model.predict_proba(feat_data_tab,model="WeightedEnsemble_L2").iloc[0,1]
        pred_list.append(pred)

    model_meta["CID_"+cid] = pred_list

    overall_predict = model_meta.groupby('set')["CID_"+cid].mean().mean()

    return overall_predict, model_meta




def main(cid, smile, p_main, p_model, p_model_meta, p_xml, p_fpmeta):
    cid = str(cid) if isinstance(cid, int) else cid  
    # for p in [p_main, p_model, p_xml, p_fpmeta]:
    p_main = p_main if isinstance(p_main, Path) else Path(p_main)
    p_model = p_model if isinstance(p_model, Path) else Path(p_model)
    p_xml = p_xml if isinstance(p_xml, Path) else Path(p_xml)
    p_fpmeta = p_fpmeta if isinstance(p_fpmeta, Path) else Path(p_fpmeta)


    print("*"*20, "Step 1/2: Prepare fingerprint feature for the input compound.", "*"*20)
    get_padel_fp(cid, smile, p_main, p_xml, p_fpmeta)
    get_padel_desc(cid, smile, p_main)

    print("*"*20, "Step 2/2: Predict the nephrotoxicity of the input compund.", "*"*20)
    pred_score, pred_table = predict_nephrotoxicity(cid, smile, p_main, p_model, p_model_meta)

    print("*"*20, f"Done! The overall estimated nephrotoxicity probality of the input compund is {pred_score:.3f}.", "*"*20)
    
    pred_score_dict = {"cid":cid, "smile":smile, "score":pred_score}


    p_save = p_main / "cid_pred" / cid
    if not p_save.exists():
        p_save.mkdir(exist_ok=True, parents=True)

    pd.DataFrame([pred_score_dict]).to_csv(p_save / "overall_score.csv", index=False)
    pred_table.to_csv(p_save / "raw_score.csv", index=False)

    return pred_score, pred_table


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--cid', type=str, required=True, help='Chemical CID')
    parser.add_argument('--smile', type=str, required=True, help='SMILE string')
    parser.add_argument('--p_main', type=str, required=True, help='Main path')
    parser.add_argument('--p_model', type=str, required=True, help='Model path')
    parser.add_argument('--p_model_meta', type=str, required=True, help='Model Metainfo path')
    parser.add_argument('--p_xml', type=str, required=True, help='Fingerprint XML path')
    parser.add_argument('--p_fpmeta', type=str, required=True, help='Fingerprint Metainfo path')

    args = parser.parse_args()

    main(args.cid, args.smile, args.p_main, args.p_model, args.p_model_meta, args.p_xml, args.p_fpmeta)


