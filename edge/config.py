from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # --- сессия ---
    talk_id: str = ""                    # id доклада из расписания
    title: str = "Демо-доклад"
    language: str | None = "ru"          # None => автодетект (осторожно при code-switching)

    # --- источник звука ---
    audio_file: str = ""                 # пусто => линейный вход с пульта
    channels: int = 1                    # каналов на входе (>1 => режим channel)
    asr_source: str = "dominant"         # dominant | mix — что кормим в ASR

    # --- ASR ---
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 1
    chunk_sec: float = 0.8               # период декодирования
    agreement_n: int = 2                 # LocalAgreement: стабильность vs +1 чанк задержки
    max_buffer_sec: float = 24.0
    # пороги отбраковки галлюцинаций (см. asr.py)
    no_speech_threshold: float = 0.65        # самостоятельное основание отбросить
    marginal_no_speech: float = 0.35         # «акустически сомнительно» для тайбрейкера
    logprob_threshold: float = -1.0          # декодер не уверен ни в чём
    compression_ratio_threshold: float = 2.4 # вырожденный самоповторяющийся текст
    boilerplate_patterns: list[str] = field(default_factory=list)   # пусто => дефолт из asr.py
    vocabulary: list[str] = field(default_factory=list)   # термины конференции (hotwords)

    # --- VAD ---
    vad: str = "energy"                  # energy | silero | off
    vad_threshold: float = 0.5
    vad_energy_db: float = -45.0
    vad_hangover_sec: float = 0.6

    # --- сегментация ---
    pause_split_sec: float = 0.6
    min_words: int = 3
    max_words: int = 18                  # страховка от спикера без точек
    clause_split_words: int = 12
    max_segment_sec: float = 3.5         # потолок ожидания перевода

    # --- определение говорящего по каналам ---
    # Пороги в dB относительно ШУМОВОГО ПОЛА КАНАЛА, не абсолютные:
    # петличка, ручной и трибунный микрофоны имеют разный гейн.
    floor_frame_sec: float = 0.032       # длина кадра анализа
    floor_open_snr: float = 8.0          # канал считается активным выше пола на столько
    floor_margin_db: float = 6.0         # отрыв лидера; меньше — считаем наложением
    floor_attack_sec: float = 0.15       # выдержка на переход слова (гасит кашель)
    floor_hold_sec: float = 0.70         # выдержка на отпускание (пауза между словами)
    floor_warmup_sec: float = 1.0        # калибровка шумового пола на старте
    floor_dead_sec: float = 90.0         # молчит дольше => микрофон выключен
    floor_noise_rise: float = 0.0003     # вверх шумовой пол ползёт еле-еле
    floor_noise_fall: float = 0.20       # к тишине падает быстро
    floor_bleed_corr: float = 0.60       # корреляция огибающих выше => просочка
    floor_corr_min_frames: int = 8

    # --- диаризация ---
    diarize_mode: str = "off"            # off | channel | embed
    diarize_channels: dict = field(default_factory=dict)   # {"0":"Мария","1":"Иван"}
    diarize_threshold: float = 0.68      # косинус, выше которого считаем «тот же голос»
    diarize_max_speakers: int = 8
    diarize_enroll_dir: str = ""
    diarize_encoder: str = "ecapa"       # см. encoders.py; eval подставляет свой
    diarize_model_dir: str = "/tmp/ecapa"

    # окно эмбеддинга. Короче 1 с ECAPA неустойчива, длиннее 2 с — окно
    # начинает захватывать двух говорящих сразу и смена голоса размазывается.
    diarize_window_sec: float = 1.5
    diarize_hop_sec: float = 0.75
    diarize_min_voiced_sec: float = 0.6  # РЕЧИ в окне после выброса пауз
    diarize_voiced_range_db: float = 35.0    # ниже максимума этого куска => пауза
    diarize_voiced_floor_db: float = -55.0   # абсолютный пол, страховка от пустого куска

    # уверенность и её пороги
    diarize_conf_temp: float = 0.06      # мягкость шкалы conf; меньше => резче
    diarize_margin_ref: float = 0.08     # отрыв от второго места, ниже которого метка спорна
    diarize_new_conf_cap: float = 0.60   # потолок для кластера с одним наблюдением
    diarize_min_conf: float = 0.45       # ниже => spk=null, чужое имя хуже пустого
    diarize_mixed_penalty: float = 0.60  # множитель, если в сегменте сменился голос

    # адаптация и склейка
    diarize_adapt_window: int = 20       # пол на скорость обучения центроида
    diarize_merge_threshold: float = 0.82    # ВЫШЕ порога назначения, см. merge_pass
    diarize_merge_every: int = 10        # склейку гоняем раз в N сегментов

    # офлайн-перекластеризация в конце доклада
    diarize_refine: bool = True
    diarize_refine_threshold: float = 0.35   # косинусное РАССТОЯНИЕ остановки

    # --- сеть ---
    cloud_ws: str = "ws://localhost:8000/ws/ingest"
    token: str = "dev-token"
    local_stage_port: int = 8788         # локальный сокет для экрана сцены

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
