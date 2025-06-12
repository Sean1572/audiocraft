# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import time
import typing as tp
import warnings

import flashy
import librosa
import math
import omegaconf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


from datasets import load_dataset
from ..data.info_audio_dataset import AudioMeta 
from ..data.music_dataset import MusicInfo 

from . import MusicGenSolver

class CLSModel(torch.nn.Module):
    def __init__(self, model, num_species):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        TODO_FIND_LAYER_SIZE = 100
        self.linear = torch.nn.Linear(TODO_FIND_LAYER_SIZE, num_species)

    def __del__(self):
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, audio_tokens, condition_tensors):
        print(self.model)
        model_output = self.model.compute_predictions(audio_tokens, [], condition_tensors)
        logits = model_output.logits
        print(logits.shape)
        out = self.linear(logits)
        return out

def one_hot(tensor, num_classes, on_value=1., off_value=0.):
    """Return one hot tensor of length num_classes
    """
    tensor = tensor.long().view(-1, 1)
    return torch.full((tensor.size()[0], num_classes), off_value, device=tensor.device) \
                .scatter_(1, tensor, on_value)

class MusicGenSolverCLS(MusicGenSolver):
    def __init__(self, cfg: omegaconf.DictConfig):
        print("MUSIC GEN CLS IS WORKING")
        super().__init__(cfg)

        ##TODO REMOVE
        # testing evaluation to make this faster.... I hope this works
        self.evaluate()

    def collate(self, batch):
        # print(batch, flush=True)
        return batch

    def transform_hugging_face_ds(self, batch):
        # print(batch)
        audios = []
        labels = []
        
        for i in range(len(batch)):
            audio, sr = librosa.load(path=batch[i]["filepath"], sr=32_000)
            label = one_hot(torch.Tensor(batch[i]["ebird_code_multilabel"]), self.num_classes)
            audios.append(torch.Tensor(audio))
            labels.append(label)

        # print(torch.vstack(audios).shape,  torch.vstack(labels).shape, flush=True)
        return torch.vstack(audios).unsqueeze(1), torch.vstack(labels)

    def build_cls_dataset(self):
        self.ds = load_dataset("DBD-research-group/BirdSet", "HSN", trust_remote_code=True)
        self.cls_dataloaders = {}
        
        for split in self.ds.keys():
            self.ds[split] = self.ds[split].with_format(None, columns=["filepath", "ebird_code_multilabel"])
            self.cls_dataloaders[split] = DataLoader(self.ds[split], batch_size=5, collate_fn=self.collate)

        self.num_classes = 21

    def build_temp_model(self):
        self.cls_model = CLSModel(self.model, self.num_classes)
        # print(self.cls_model)
        
    def get_metadata(self, filepath):
        return MusicInfo(
                AudioMeta(
                    path=filepath,
                    duration=5,
                    sample_rate=32_000,
                ),
                seek_time=0,
                n_frames=32_000 * 5,
                total_frames=32_000 * 5,
                channels=1,
                sample_rate=32_000
            )
    def evaluate_cls_application(self):
        """Evaluation but for CLS""" 

        #Step 1: Get Dataset
        self.build_cls_dataset()

        #Step 2: Get a model prepped
        self.build_temp_model()

        # Short training loop, should only do an epoch as a quick test
        # TODO

        # Evaluation Loop
        for i, batch in enumerate(self.cls_dataloaders["test_5s"]):
            # Step 3: Data preprocessing to fake a dataset, because I am lazy
            # TODO: turn this into a function call, need for training too
            audio_data, labels = self.transform_hugging_face_ds(batch)
            condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(
                (audio_data, [self.get_metadata(item["filepath"]) for item in batch]))

            # TODO Get metrics from model performance on soundscapes / xc validation split?
            
            print(audio_tokens.dtype, condition_tensors)
            self.cls_model(audio_tokens, condition_tensors)

        del self.cls_model

        print("works!")
        print("works!")

        print("works!")
        print("works!")
        print("works!")
        print("works!")
        print("works!")
        print("works!")


        print("works!")
        print("works!")
        print("works!")
        print("works!")
        print("works!")
        print("works!")
        return {"Test_Metric_FOR_CLS": [100]}


    ## Overrides the following methods to run above system
    def evaluate(self):
        generative_results = {} #super().evaluate() #Required being in a stage, so needs more from system
        self.model.eval()
        with torch.no_grad():
            cls_results = self.evaluate_cls_application()
        generative_results.update(cls_results)
        return generative_results