from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    
    talk_id: str = ""                    
    title: str = "Демо-доклад"
    language: str | None = "ru"          

    audio_file: str = ""                 

    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 1
    chunk_sec: float = 0.8               
    agreement_n: int = 2                 
    max_buffer_sec: float = 24.0
    no_speech_threshold: float = 0.65
    vocabulary: list[str] = field(default_factory=list)   

    vad: str = "energy"                  
    vad_threshold: float = 0.5
    vad_energy_db: float = -45.0
    vad_hangover_sec: float = 0.6

   
    pause_split_sec: float = 0.6
    min_words: int = 3
    max_words: int = 18                  
    clause_split_words: int = 12
    max_segment_sec: float = 3.5         

   
    diarize_mode: str = "off"            
    diarize_channels: dict = field(default_factory=dict)   
    diarize_threshold: float = 0.68
    diarize_max_speakers: int = 8
    diarize_enroll_dir: str = ""

    
    cloud_ws: str = "ws://localhost:8000/ws/ingest"
    token: str = "dev-token"
    local_stage_port: int = 8788        

    @staticmethod
    def load(path: str | None) -> "Config":
        cfg = Config()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = yaml_or_json(f.read())
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


def yaml_or_json(text: str) -> dict:
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        return json.loads(text)
