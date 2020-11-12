import os
from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops, Draw
from rdkit.Chem import MolFromSmiles

def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                          nBits=size,
                                                          useChirality=True,
                                                          useBondTypes=True,
                                                          useFeatures=False
                                                          )

def generate_all_images(X_train, y_train, X_test, y_test,
    img_dir="../data/smile_images", target_size=(224, 224)):
    print("Generating all molecule images")

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    try: os.makedirs(os.path.join(img_dir, "train", "0"))
    except: pass
    try: os.makedirs(os.path.join(img_dir, "train", "1"))
    except: pass
    try: os.makedirs(os.path.join(img_dir, "test", "0"))
    except: pass
    try: os.makedirs(os.path.join(img_dir, "test", "1"))
    except: pass

    for idx, smile_str in enumerate(X_train):
        if idx % 1000 == 0:
            print("Generating img {}".format(idx))
        molsmile = MolFromSmiles(smile_str)
        prop_P1 = y_train[idx]
        Draw.MolToFile(molsmile, '{}/train/{}/{}.png'.format(img_dir, prop_P1,
            smile_str.replace("/", "-")), size = target_size)
    for idx, smile_str in enumerate(X_test):
        if idx % 1000 == 0:
            print(idx)
        molsmile = MolFromSmiles(smile_str)
        prop_P1 = y_test[idx]
        Draw.MolToFile(molsmile, '{}/test/{}/{}.png'.format(img_dir, prop_P1,
            smile_str.replace("/", "-")), size = target_size)
    print("Finished creating images")
