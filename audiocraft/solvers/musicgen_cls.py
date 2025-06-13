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
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader


from datasets import load_dataset
from ..data.info_audio_dataset import AudioMeta
from ..data.music_dataset import MusicInfo, WavCondition  

from . import MusicGenSolver

class CLSModel(nn.Module):
    def __init__(self, model, num_species):
        super().__init__()
        
        self.model = model
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # All dimensions that aren't batch for final classification
        # Does this make sense? Probably not. Who is to say the last layer of music gen is useful at all!
        # Will need to dig deeper into musicgen to figure out most useful layer
        self.FINAL_LAYER_SIZE = 2048 * 250 * 4 
        self.linear = nn.Linear(self.FINAL_LAYER_SIZE, num_species, dtype=torch.float32)
    
    def forward(self, audio_tokens, condition_tensors):
        with torch.no_grad():
            model_output = self.model(audio_tokens, [], condition_tensors) #.compute_predictions
            logits = model_output.reshape(-1, self.FINAL_LAYER_SIZE)

        logits = torch.Tensor(logits.cpu().detach().cuda()).requires_grad_()
        out = self.linear(logits)

        return out

def one_hot(tensor, num_classes, on_value=1., off_value=0.):
    """Return one hot tensor of length num_classes
    """
    tensor = tensor.long().view(-1, 1)
    return torch.clamp(
        torch.full((tensor.size()[0], num_classes), off_value, device=tensor.device) \
                .scatter_(1, tensor, on_value).sum(dim=0),
        min=0, max=1)

class MusicGenSolverCLS(MusicGenSolver):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)

        ##TODO REMOVE
        # testing evaluation to make this faster.... I hope this works
        self.evaluate()

    def collate(self, batch):
        return batch

    def transform_hugging_face_ds(self, batch):
        audios = []
        labels = []
        
        for i in range(len(batch)):
            y, sr = librosa.load(path=batch[i]["filepath"], sr=32_000)
            label = one_hot(torch.Tensor(batch[i]["ebird_code_multilabel"]), self.num_classes)
            start = 0

            # Select a random 5 second segment
            # From https://github.com/UCSD-E4E/pyha-analyzer-2.0/blob/main/pyha_analyzer/preprocessors/spectogram_preprocessors.py
            if y.shape[-1] > (sr * 5):
                start = np.random.randint(0, y.shape[-1] - (sr * 5))
            else:
                y = np.pad(y, (sr * 5) - y.shape[-1])
            audios.append(torch.Tensor(y[start : start + (sr * 5)]))
            labels.append(label)

        return torch.vstack(audios).unsqueeze(1).cuda(), torch.vstack(labels).cuda()

    def build_cls_dataset(self):
        self.ds = load_dataset("DBD-research-group/BirdSet", "HSN", trust_remote_code=True)
        self.cls_dataloaders = {}
        
        for split in self.ds.keys():
            self.ds[split] = self.ds[split].with_format(None, columns=["filepath", "ebird_code_multilabel"])
            self.cls_dataloaders[split] = DataLoader(self.ds[split], batch_size=1, collate_fn=self.collate)

        self.num_classes = 21

    def build_temp_model(self):
        cls_model = CLSModel(self.model, self.num_classes)
        
    def get_metadata(self, filepath, audio=None):
        music_info = MusicInfo(
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
        if audio is not None:
            music_info.self_wav = WavCondition(
                wav=audio, length=torch.tensor([music_info.n_frames]),
                sample_rate=[music_info.sample_rate], path=[music_info.meta.path], seek_time=[music_info.seek_time])
            
        return music_info
    
    def evaluate_cls_application(self):
        """Evaluation but for CLS""" 

        # For some reason, grad isn't enabled by default...
        torch.set_grad_enabled(True)

        #Step 1: Get Dataset
        self.build_cls_dataset()

        #Step 2: Get a model prepped
        #self.build_temp_model()
        cls_model = CLSModel(self.model, self.num_classes)
        cls_model = cls_model.cuda()

        # Train Loop
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(cls_model.parameters(), lr=0.001)
        
        cls_model = cls_model.train()
        for i, batch in enumerate(self.cls_dataloaders["train"]):
            # Step 3: Data preprocessing to fake a dataset, because I am lazy
            # TODO: turn this into a function call, need for training too
            audio_data, labels = self.transform_hugging_face_ds(batch)
            condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(
                (
                    audio_data, 
                    [
                        self.get_metadata(
                            batch[i]["filepath"], 
                            audio=audio_data[i]
                        ) for i in range(len(batch))
                    ]
                )
            )
            audio_tokens = audio_tokens
            description_tesnor = condition_tensors["description"][0].to(torch.float32)
            condition_tensors["description"] = (description_tesnor, condition_tensors["description"][1])

            optimizer.zero_grad()
            output = cls_model(audio_tokens, condition_tensors)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            break #TODO REMOVE

        # Evaluation Loop
        cls_model.eval()
        for i, batch in enumerate(self.cls_dataloaders["test_5s"]):
            # Step 3: Data preprocessing to fake a dataset, because I am lazy
            # TODO: turn this into a function call, need for training too
            audio_data, labels = self.transform_hugging_face_ds(batch)
            condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(
                (audio_data, [self.get_metadata(item["filepath"]) for item in batch]))
            description_tesnor = condition_tensors["description"][0].to(torch.float32)
            condition_tensors["description"] = (description_tesnor, condition_tensors["description"][1])

            # TODO Get metrics from model performance on soundscapes / xc validation split?
            output = cls_model(audio_tokens, condition_tensors)
            loss = criterion(output, labels)
            print(loss) #Collect metrics here
            break #TODO REMOVE

        del cls_model
        return {"Test_Metric_FOR_CLS": [100]} #TODO actually collect metrics


    ## Overrides the following methods to run above system
    def evaluate(self):
        generative_results = {} #super().evaluate() #Required being in a stage, so needs more from system
        self.model.eval()
        with torch.no_grad():
            cls_results = self.evaluate_cls_application()
        generative_results.update(cls_results)
        return generative_results