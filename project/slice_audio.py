from typing import Any
import json
import pathlib
import random
from typing import Union

import torchaudio
import torch
import numpy as np
import pandas as pd

from utils import time_to_sample_conv



class Audio_File:
    
    def __init__(self, path: str) -> None:
        self.path = path
        self.waveform, self.sample_rate = torchaudio.load(self.path)
        self.metadata = torchaudio.info(self.path)
            
                           
class Annotations_File:
    
    def __init__(self, path: str) -> None:
        self.path = path
        self.annos = self._load_annos()   
        
    def _load_annos(self) -> list[dict[str, Any]]:
        with open(self.path, 'r') as f:
            json_object = json.load(f)
        
        return json_object['audio_spans']
        
        

class Slicer:
    
     
    def __init__(self, audio: Audio_File, annotations: Annotations_File, label2id: dict) -> None:
        self.audio = audio
        self.annotations = annotations
        self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.on_samples = self._create_spans()
        self.train_index = 1
        self.test_index = 1
        self.TRAIN_DIR = pathlib.Path('dataset', 'train')
        self.TEST_DIR = pathlib.Path('dataset', 'test') 
       
        
    def __len__(self):
        return len(self.on_samples)
    
    
    def _create_spans(self): 
        spans = torch.zeros_like(self.audio.waveform)
        for dict in self.annotations.annos:
            tuple_of_secs = tuple(v for k, v in dict.items() if k in ['start', 'end'])
            # Does sample_rate need to be a variable?
            span_tensor = time_to_sample_conv(*tuple_of_secs, sample_rate=16000, span=False)
            start_inx = span_tensor[0].item()
            stop_inx = span_tensor[-1].item()
            spans[0, start_inx:stop_inx] = 1
        return spans
    
    
    def slicer(self, dir: Union[str, pathlib.Path], count: float, units:str='seconds') -> None:
        dir = pathlib.Path(dir)
        self._create_file_structure_if_neccessary(dir)
        
        if units == "seconds":
            frames_to_slice = int(count * 16000)
        elif units == "samples":
            frames_to_slice = int(count)
        else:
            raise Exception('please select either "seconds" or "samples"')
        
        csv_container:list[Any] = []
        
        for start_fr in range(0, self.audio.metadata.num_frames, frames_to_slice):
            stop_fr = start_fr + frames_to_slice
            label = self._check_for_label(start_fr, stop_fr) 
            snippet_path = self._create_snippet_path(dir, self.TRAIN_DIR, self.TEST_DIR)
            self._save_audio_snippet(start_fr, stop_fr, snippet_path)
            self._add_metadata(dir, snippet_path, label, csv_container)
        
        metadata_filename = str(pathlib.Path(dir, 'dataset', 'metadata.csv'))
        pd.DataFrame(csv_container).to_csv(metadata_filename, index=False)   
    
    
    def _create_file_structure_if_neccessary(self, dir: pathlib.Path) -> None:
        abs_train_path = pathlib.Path(dir, self.TRAIN_DIR)
        abs_test_path = pathlib.Path(dir, self.TEST_DIR)
        
        if not abs_train_path.is_dir() :
           abs_train_path.mkdir(parents=True)
        if not abs_test_path.is_dir():
            abs_test_path.mkdir(parents=True)
                    
            
    def _check_for_label(self, start: int, stop: int) -> str:
        if np.argmax(self.on_samples[0, start:stop]):
            return self.id2label[1]
        else:
            return self.id2label[0]
        
        
    def _create_snippet_path(self,dir:pathlib.Path, train_dir:pathlib.Path, test_dir:pathlib.Path) -> pathlib.Path:
        if random.randint(0, 9) < 8:
            index_and_ending = f'{str(self.train_index)}.wav'
            path = pathlib.Path(dir, train_dir, index_and_ending)
            self.train_index = self.train_index + 1
            return path
        else:
            index_and_ending = f'{str(self.test_index)}.wav'
            path = pathlib.Path(dir, test_dir, index_and_ending)
            self.test_index = self.test_index + 1
            return path
        
    
    def _save_audio_snippet(self, start:int, stop:int, snippet_path:pathlib.Path) -> None:
        snippet = self.audio.waveform[0, start:stop].unsqueeze(0)
        torchaudio.save(filepath=str(snippet_path),
                        src=snippet,
                        sample_rate=16000,
                        format='wav',
                        encoding='PCM_S',
                        bits_per_sample=16
                        )
        
                
    def _add_metadata(self,dir:pathlib, snippet_path:pathlib.Path, label:str, csv_container:list[Any]) -> None:
        comparison_path = pathlib.Path(dir, 'dataset')
        relative_path = snippet_path.relative_to(comparison_path)
        file_and_label = {'file_name': str(relative_path), 'label':label2id[label]}
        csv_container.append(file_and_label)
        
        
                
if __name__ == "__main__":
    
    video_1_audio = Audio_File("/home/jbjoyce/transformers-course/project/audio_files/1.wav")
    video_1_annos = Annotations_File("/home/jbjoyce/transformers-course/project/annotations/1.json1")
    label2id = {'OTHER':0, 'DENTAL_CLICK':1}
    
    dental_click_slicer = Slicer(video_1_audio, video_1_annos, label2id)
    dental_click_slicer.slicer('/home/jbjoyce/transformers-course/project', 1, 'seconds')
    
     
         
        
        