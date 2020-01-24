"""
actam:
    python data_download.py smiles -p data/datasetName


Downloading QM9/ DECAGON data to args.path (set by default to be "./data/")
command: python data_download QM9 DECAGON -p path
"""
import argparse
import os
import wget
import zipfile
import tarfile
from utils.file_utils import *


def download_smiles_data(dir_path='./data/'):
    """
    Step 0: Download Polypharmacy data
    wget http://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz;
    tar -xvzf bio-decagon-combo.tar.gz;

    bio-decagon-combo.csv which has the form
    drug_CID, drug_CID, side_effect_id, side_effect_name
    ( == CIDXXXXX,CIDXXXX,CXXXX,side_effect_name).

    Step 1: Collect drug cid list in drug_raw_feat.idx.jsonl,
    which has the form:
        CIDXXXX : { "atoms": [
                        {"aid": x, "number": x, "x": x, "y": x}
                        ... ]
                    "bonds": [
                        {"aid1": x, "aid2": x, "order": x}
                        ... ]
                    }
    where
        atom.aid is the atom ID within the owning Compound (molecule)
        atom.number is the atomic number for this atom
        atom.x, atom.y are the coordinates

        bond.aid1, bond.aid2 are the begin and end atom of the bond
        bond.order is the (chemical) bond order
    For more details
        https://pubchempy.readthedocs.io/en/latest/api.html#pubchempy.Atom
    """
    prepare_data_dir(dir_path)

    import csv
    drug_idx = set()
    with open(os.path.join(dir_path, 'data.csv')) as f:
        csv_rdr = csv.reader(f)
        for i, row in enumerate(csv_rdr):
            if i == 0:
                print('Header:', row)
            else:
                drug1, drug2, *_ = row
                drug_idx |= {drug1, drug2}
    print('Instance:', row)

    print('Unique drug count =', len(drug_idx))

    # # Step 2: Search on PubChem
    from tqdm import tqdm
    import pubchempy as pcp
    # Use int type cid to search with PubChemPy
    drugs = {cid: pcp.get_compounds(cid, 'smiles')[0]
             for cid in tqdm(drug_idx)}

    # # Step 3: Write to file
    import json
    with open(os.path.join(dir_path, 'drug_raw_feat.idx.jsonl'), 'w') as f:
        for cid, drug in drugs.items():
            drug = drug.to_dict(properties=['atoms', 'bonds'])
            f.write('{}\t{}\n'.format(cid, json.dumps(drug)))


def main():
    parser = argparse.ArgumentParser(
        description='Download dataset for Graph Co-attention')

    # I/O
    parser.add_argument('-p', '--path', metavar='dir', type=str, nargs=1,
                        help="path to store the data", required=True)

    args = parser.parse_args()

    # Check parameters
    if args.path is None:
        args.path = './data/'
    else:
        args.path = args.path[0]

    prepare_data_dir(args.path)
    download_smiles_data(args.path)


if __name__ == "__main__":
    main()
