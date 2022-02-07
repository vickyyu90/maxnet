
import numpy as np
import argparse
import os
import abc
from os import path
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda

FILENAME_TYPE = {'full': '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w',
                 'cropped': '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w',
                 'skull_stripped': '_space-Ixi549Space_desc-skullstripped_T1w'}

class ToTensor(object):
    """Convert image type to Tensor and diagnosis to diagnosis code"""

    def __call__(self, image):
        np.nan_to_num(image, copy=False)
        image = image.astype(float)

        return torch.from_numpy(image[np.newaxis, :]).float()


class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())

def load_data(train_val_path, diagnoses_list, baseline=True):

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    test_df = pd.DataFrame()

    train_path = path.join(train_val_path, 'train')
    valid_path = path.join(train_val_path, 'validation')
    test_path = path.join(train_val_path, 'test')

    for diagnosis in diagnoses_list:

        if baseline:
            train_diag_path = path.join(
                train_path, diagnosis + '_baseline.tsv')
        else:
            train_diag_path = path.join(train_path, diagnosis + '.tsv')

        valid_diag_path = path.join(valid_path, diagnosis + '_baseline.tsv')
        test_diag_path = path.join(test_path, diagnosis + '_baseline.tsv')

        train_diag_df = pd.read_csv(train_diag_path, sep='\t')
        valid_diag_df = pd.read_csv(valid_diag_path, sep='\t')
        test_diag_df = pd.read_csv(test_diag_path, sep='\t')

        train_df = pd.concat([train_df, train_diag_df])
        valid_df = pd.concat([valid_df, valid_diag_df])
        test_df = pd.concat([test_df, test_diag_df])

    train_df.reset_index(inplace=True, drop=True)
    valid_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)

    return train_df, valid_df, test_df

def get_transforms(mode, minmaxnormalization=True):
    if mode in ["image", "patch", "roi"]:
        if minmaxnormalization:
            transformations = MinMaxNormalization()
        else:
            transformations = None
    elif mode == "slice":
        trg_size = (224, 224)
        if minmaxnormalization:
            transformations = transforms.Compose([MinMaxNormalization(),
                                                  transforms.ToPILImage(),
                                                  transforms.Resize(trg_size),
                                                  transforms.ToTensor()])
        else:
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(trg_size),
                                                  transforms.ToTensor()])
    else:
        raise ValueError("Transforms for mode %s are not implemented." % mode)

    return transformations

def return_dataset(mode, input_dir, data_df, preprocessing,
                   transformations, params, cnn_index=None):
    if mode == "roi":
        return MRIDatasetRoi(
            input_dir,
            data_df,
            preprocessing=preprocessing,
            transformations=transformations
        )

def get_dataloaders(args: argparse.Namespace):
    """
    Get data loaders
    """
    # Obtain the dataset
    transformations = get_transforms(args.mode, not args.unnormalize)

    training_df, valid_df, test_df = load_data(
        args.tsv_path,
        args.diagnoses,
        baseline=args.baseline)

    data_train = return_dataset(args.mode, args.caps_dir, training_df, args.preprocessing,
                                transformations, args)
    data_valid = return_dataset(args.mode, args.caps_dir, valid_df, args.preprocessing,
                                transformations, args)
    data_test = return_dataset(args.mode, args.caps_dir, test_df, args.preprocessing,
                                transformations, args)

    trainloader = torch.utils.data.DataLoader(
        data_train,
        batch_size= args.batch_size,
        shuffle=True,
        num_workers= args.nproc,
        pin_memory=True
    )

    testloader = torch.utils.data.DataLoader(
        data_test,
        batch_size= args.batch_size,
        shuffle=True,
        num_workers= args.nproc,
        pin_memory=True
    )

    valloader = torch.utils.data.DataLoader(
        data_valid,
        batch_size= args.batch_size,
        shuffle=False,
        num_workers= args.nproc,
        pin_memory=True
    )

    return trainloader, valloader, testloader


class MRIDataset(Dataset):
    def __init__(self, caps_directory, data_file,
                 preprocessing, transformations=None):
        self.caps_directory = caps_directory
        self.transformations = transformations
        self.diagnosis_code = {
            'CN': 0,
            'AD': 1,
            'sMCI': 0,
            'pMCI': 1,
            'MCI': 1,
            'unlabeled': -1}
        self.preprocessing = preprocessing

        if not hasattr(self, 'elem_index'):
            raise ValueError(
                "Child class of MRIDataset must set elem_index attribute.")
        if not hasattr(self, 'mode'):
            raise ValueError(
                "Child class of MRIDataset must set mode attribute.")

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument data_file is not of correct type.')

        mandatory_col = {"participant_id", "session_id", "diagnosis"}
        if self.elem_index == "mixed":
            mandatory_col.add("%s_id" % self.mode)

        if not mandatory_col.issubset(set(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include %s" % mandatory_col)

        self.elem_per_image = self.num_elem_per_image()

    def __len__(self):
        return len(self.df) * self.elem_per_image

    def _get_path(self, participant, session, mode="image"):

        if self.preprocessing == "t1-linear":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_linear',
                                   participant + '_' + session
                                   + FILENAME_TYPE['cropped'] + '.pt')
        elif self.preprocessing == "t1-extensive":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_extensive',
                                   participant + '_' + session
                                   + FILENAME_TYPE['skull_stripped'] + '.pt')
        else:
            raise NotImplementedError(
                "The path to preprocessing %s is not implemented" % self.preprocessing)

        return image_path

    def _get_meta_data(self, idx):
        image_idx = idx // self.elem_per_image
        participant = self.df.loc[image_idx, 'participant_id']
        session = self.df.loc[image_idx, 'session_id']

        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        elif self.elem_index == "mixed":
            elem_idx = self.df.loc[image_idx, '%s_id' % self.mode]
        else:
            elem_idx = self.elem_index

        diagnosis = self.df.loc[image_idx, 'diagnosis']
        label = self.diagnosis_code[diagnosis]

        return participant, session, elem_idx, label

    def _get_full_image(self):
        import nibabel as nib

        participant_id = self.df.loc[0, 'participant_id']
        session_id = self.df.loc[0, 'session_id']

        try:
            image_path = self._get_path(participant_id, session_id, "image")
            image = torch.load(image_path)
        except FileNotFoundError:
            image_path = find_image_path(
                self.caps_directory,
                participant_id,
                session_id,
                preprocessing=self.preprocessing)
            image_nii = nib.load(image_path)
            image_np = image_nii.get_fdata()
            image = ToTensor()(image_np)

        return image

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    @abc.abstractmethod
    def num_elem_per_image(self):
        pass

def find_image_path(caps_dir, participant_id, session_id, preprocessing):
    from os import path
    if preprocessing == "t1-linear":
        image_path = path.join(caps_dir, 'subjects', participant_id, session_id,
                               't1_linear',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['cropped'] + '.nii.gz')
    elif preprocessing == "t1-extensive":
        image_path = path.join(caps_dir, 'subjects', participant_id, session_id,
                               't1', 'spm', 'segmentation', 'normalized_space',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['skull_stripped'] + '.nii.gz')
    else:
        raise ValueError(
            "Preprocessing %s must be in ['t1-linear', 't1-extensive']." %
            preprocessing)

    return image_path


class MRIDatasetRoi(MRIDataset):

    def __init__(self, caps_directory, data_file, preprocessing="t1-linear",
                 transformations=None, prepare_dl=False):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            transformations (callable, optional): Optional transform to be applied on a sample.
            prepare_dl (bool): If true pre-extracted patches will be loaded.

        """
        self.elem_index = None
        self.mode = "roi"
        super().__init__(caps_directory, data_file, preprocessing, transformations)
        self.prepare_dl = prepare_dl

    def __getitem__(self, idx):
        participant, session, roi_idx, label = self._get_meta_data(idx)
        if self.prepare_dl:
            raise NotImplementedError(
                'The extraction of ROIs prior to training is not implemented.')

        else:
            image_path = self._get_path(participant, session, "image")
            image = torch.load(image_path)
            #patch = self.extract_roi_from_mri(image, roi_idx)

        if self.transformations:
            patch = self.transformations(image)

        sample = {'image': image, 'label': label,
                  'participant_id': participant, 'session_id': session,
                  'roi_id': roi_idx}

        return sample

    def num_elem_per_image(self):
        return 2

    def extract_roi_from_mri(self, image_tensor, left_is_odd):
        if self.preprocessing == "t1-linear":
            if left_is_odd == 1:
                # the center of the left hippocampus
                crop_center = (61, 96, 68)
            else:
                # the center of the right hippocampus
                crop_center = (109, 96, 68)
        else:
            raise NotImplementedError("The extraction of hippocampi was not implemented for "
                                      "preprocessing %s" % self.preprocessing)
        crop_size = (50, 50, 50)  # the output cropped hippocampus size

        extracted_roi = image_tensor[
            :,
            crop_center[0] - crop_size[0] // 2: crop_center[0] + crop_size[0] // 2:,
            crop_center[1] - crop_size[1] // 2: crop_center[1] + crop_size[1] // 2:,
            crop_center[2] - crop_size[2] // 2: crop_center[2] + crop_size[2] // 2:
        ].clone()

        return extracted_roi


