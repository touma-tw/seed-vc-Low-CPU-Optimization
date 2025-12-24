import os
import sys
# [核心優化] 鎖定執行緒，防止 Windows CPU 資源搶奪
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dotenv import load_dotenv
import shutil
import warnings
import yaml
import time
import json
import re
import argparse
import librosa
import numpy as np
import FreeSimpleGUI as sg
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as tat
from modules.commons import *
from hf_utils import load_custom_model_from_hf

load_dotenv()
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

warnings.simplefilter("ignore")
now_dir = os.getcwd()
sys.path.append(now_dir)

# --- 全域變數 ---
device = None
flag_vc = False
prompt_condition, mel2, style2 = None, None, None
reference_wav_name = ""
prompt_len = 3
ce_dit_difference = 2.0
fp16 = False

@torch.no_grad()
def custom_infer(model_set, reference_wav, new_reference_wav_name, input_wav_res,
                 block_frame_16k, skip_head, skip_tail, return_length,
                 diffusion_steps, inference_cfg_rate, max_prompt_length,
                 cd_difference=2.0):
    global prompt_condition, mel2, style2, reference_wav_name, prompt_len, ce_dit_difference
    (model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args) = model_set
    sr = mel_fn_args["sampling_rate"]
    hop_length = mel_fn_args["hop_size"]

    if ce_dit_difference != cd_difference: ce_dit_difference = cd_difference
    
    if prompt_condition is None or reference_wav_name != new_reference_wav_name or prompt_len != max_prompt_length:
        prompt_len = max_prompt_length
        reference_wav = reference_wav[:int(sr * prompt_len)]
        reference_wav_tensor = torch.from_numpy(reference_wav).to(device)
        ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
        S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
        feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))
        mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        prompt_condition = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=None)[0]
        reference_wav_name = new_reference_wav_name

    S_alt = semantic_fn(input_wav_res.unsqueeze(0))
    ce_dit_frame = int(ce_dit_difference * 50)
    S_alt = S_alt[:, ce_dit_frame:]
    target_lengths = torch.LongTensor([(skip_head + return_length + skip_tail - ce_dit_frame) / 50 * sr // hop_length]).to(S_alt.device)
    cond = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)
    
    with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
        vc_target = model.cfm.inference(
            cat_condition, torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2, style2, None, n_timesteps=diffusion_steps, inference_cfg_rate=inference_cfg_rate
        )
        vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target).squeeze()
        
    output_len = return_length * sr // 50
    tail_len = skip_tail * sr // 50
    
    # [核心] 強制轉 FP32
    return vc_wave[-output_len - tail_len: -tail_len].float()

def load_models(args):
    global fp16
    fp16 = args.fp16
    if args.checkpoint_path is None or args.checkpoint_path == "":
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC", "DiT_uvit_tat_xlsr_ema.pth", "config_dit_mel_seed_uvit_xlsr_tiny.yml")
    else:
        dit_checkpoint_path = args.checkpoint_path
        dit_config_path = args.config_path
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    sr = config["preprocess_params"]["sr"]
    model, _, _, _ = load_checkpoint(model, None, dit_checkpoint_path, load_only_params=True, ignore_modules=[], is_distributed=False)
    for key in model: model[key].eval().to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    from modules.campplus.DTDNN import CAMPPlus
    campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval().to(device)

    # Vocoder Loader
    vocoder_type = model_params.vocoder.type
    vocoder_fn = None
    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        try: v_name = model_params.vocoder.name
        except: v_name = "nvidia/bigvgan_v2_22khz_80band_256x"
        try: vocoder_fn = bigvgan.BigVGAN.from_pretrained(v_name, use_cuda_kernel=False)
        except: vocoder_fn = None
        if vocoder_fn is None: raise RuntimeError(f"Could not load BigVGAN: {v_name}")
        vocoder_fn.remove_weight_norm().eval().to(device)
    elif vocoder_type == 'hifigan':
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        try:
            hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
            hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
            hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
            hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
            hift_gen.eval().to(device)
            vocoder_fn = hift_gen
        except: raise RuntimeError("Failed to load HiFiGAN")
    elif vocoder_type == "vocos":
        try:
            vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
            vocos = build_model(recursive_munch(vocos_config['model_params']), stage='mel_vocos')
            vocos, _, _, _ = load_checkpoint(vocos, None, model_params.vocoder.vocos.path, load_only_params=True, ignore_modules=[], is_distributed=False)
            for key in vocos: vocos[key].eval().to(device)
            vocoder_fn = vocos.decoder
        except: raise RuntimeError("Failed to load Vocos")
    else: raise ValueError(f"Unknown vocoder: {vocoder_type}")

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    model_name = config['model_params']['speech_tokenizer']['name']
    output_layer = config['model_params']['speech_tokenizer']['output_layer']
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
    wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
    wav2vec_model = wav2vec_model.to(device).eval().half()

    def semantic_fn(waves_16k):
        ori_inputs = wav2vec_feature_extractor([waves_16k[0].cpu().numpy()], return_tensors="pt", sampling_rate=16000).to(device)
        with torch.no_grad(): ori_outputs = wav2vec_model(ori_inputs.input_values.half())
        return ori_outputs.last_hidden_state.float()

    mel_fn_args = {"n_fft": 1024, "win_size": 1024, "hop_size": 256, "num_mels": 80, "sampling_rate": sr, "fmin": 0, "fmax": 8000, "center": False}
    from modules.audio import mel_spectrogram
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
    return (model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args)

def printt(strr, *args):
    if len(args) == 0: print(strr)
    else: print(strr % args)

class Config:
    def __init__(self): self.device = device

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--fp16", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda"
    device = torch.device(cuda_target if torch.cuda.is_available() else "cpu")

    class GUIConfig:
        def __init__(self) -> None:
            self.reference_audio_path: str = ""
            self.diffusion_steps: int = 10
            self.sr_type: str = "sr_model"
            self.block_time: float = 0.5
            self.threhold: int = -50
            self.crossfade_time: float = 0.05
            self.extra_time_ce: float = 2.5
            self.extra_time: float = 0.5
            self.extra_time_right: float = 2.0
            self.I_noise_reduce: bool = False
            self.O_noise_reduce: bool = False
            self.inference_cfg_rate: float = 0.7
            self.sg_hostapi: str = ""
            self.wasapi_exclusive: bool = False
            self.sg_input_device: str = ""
            self.sg_output_device: str = ""
            self.samplerate: int = 48000
            self.mic_gain: float = 1.0 # [新增]

    class GUI:
        def __init__(self, args) -> None:
            self.gui_config = GUIConfig()
            self.config = Config()
            self.function = "vc"
            self.delay_time = 0
            self.hostapis = None
            self.input_devices = None
            self.output_devices = None
            self.stream = None
            self.model_set = load_models(args)
            self.vad_speech_detected = False
            self.update_devices()
            self.launcher()

        def load(self):
            try:
                os.makedirs("configs/inuse", exist_ok=True)
                if not os.path.exists("configs/inuse/config.json"):
                    shutil.copy("configs/config.json", "configs/inuse/config.json")
                with open("configs/inuse/config.json", "r") as j:
                    data = json.load(j)
                    data["sr_model"] = data["sr_type"] == "sr_model"
                    data["sr_device"] = data["sr_type"] == "sr_device"
            except: data = {}
            return data

        def launcher(self):
            self.config = Config()
            data = self.load()
            sg.theme("LightBlue3")
            layout = [
                [sg.Frame(title="Load reference audio", layout=[[sg.Input(default_text=data.get("reference_audio_path", ""), key="reference_audio_path"), sg.FileBrowse("choose an audio file", initial_folder=os.path.join(os.getcwd(), "wav"), file_types=[("WAV Files", "*.wav"), ("MP3 Files", "*.mp3")])]])],
                [sg.Frame(layout=[
                    [sg.Text("Device type"), sg.Combo(self.hostapis, key="sg_hostapi", default_value=data.get("sg_hostapi", ""), enable_events=True, size=(20, 1)), sg.Checkbox("WASAPI Exclusive Device", key="sg_wasapi_exclusive", default=data.get("sg_wasapi_exclusive", False), enable_events=True)],
                    [sg.Text("Input Device"), sg.Combo(self.input_devices, key="sg_input_device", default_value=data.get("sg_input_device", ""), enable_events=True, size=(45, 1))],
                    [sg.Text("Output Device"), sg.Combo(self.output_devices, key="sg_output_device", default_value=data.get("sg_output_device", ""), enable_events=True, size=(45, 1))],
                    [sg.Button("Reload devices", key="reload_devices"), sg.Radio("Use model sampling rate", "sr_type", key="sr_model", default=True, enable_events=True), sg.Radio("Use device sampling rate", "sr_type", key="sr_device", default=False, enable_events=True), sg.Text("Sampling rate:"), sg.Text("", key="sr_stream")]
                ], title="Sound Device")],
                [sg.Frame(layout=[
                    [sg.Text("Activation threshold"), sg.Slider(range=(-60, 0), key="threhold", resolution=1, orientation="h", default_value=data.get("threhold", -50), enable_events=True)],
                    [sg.Text("Diffusion steps"), sg.Slider(range=(1, 30), key="diffusion_steps", resolution=1, orientation="h", default_value=data.get("diffusion_steps", 10), enable_events=True)],
                    [sg.Text("Inference cfg rate"), sg.Slider(range=(0.0, 1.0), key="inference_cfg_rate", resolution=0.1, orientation="h", default_value=data.get("inference_cfg_rate", 0.7), enable_events=True)],
                    [sg.Text("Max prompt length (s)"), sg.Slider(range=(1.0, 20.0), key="max_prompt_length", resolution=0.5, orientation="h", default_value=data.get("max_prompt_length", 3.0), enable_events=True)],
                    # [新增] Mic Gain 拉桿
                    [sg.Text("Mic Gain (x)"), sg.Slider(range=(0.3, 5.0), key="mic_gain", resolution=0.1, orientation="h", default_value=1.0, enable_events=True)],
                ], title="Regular settings"),
                sg.Frame(layout=[
                    [sg.Text("Block time"), sg.Slider(range=(0.04, 3.0), key="block_time", resolution=0.02, orientation="h", default_value=data.get("block_time", 0.5), enable_events=True)],
                    [sg.Text("Crossfade length"), sg.Slider(range=(0.02, 0.5), key="crossfade_length", resolution=0.02, orientation="h", default_value=data.get("crossfade_length", 0.1), enable_events=True)],
                    [sg.Text("Extra content encoder context time (left)"), sg.Slider(range=(0.5, 10.0), key="extra_time_ce", resolution=0.1, orientation="h", default_value=data.get("extra_time_ce", 5.0), enable_events=True)],
                    [sg.Text("Extra DiT context time (left)"), sg.Slider(range=(0.5, 10.0), key="extra_time", resolution=0.1, orientation="h", default_value=data.get("extra_time", 5.0), enable_events=True)],
                    [sg.Text("Extra context time (right)"), sg.Slider(range=(0.02, 10.0), key="extra_time_right", resolution=0.02, orientation="h", default_value=data.get("extra_time_right", 2.0), enable_events=True)]
                ], title="Performance settings")],
                [sg.Button("Start Voice Conversion", key="start_vc"), sg.Button("Stop Voice Conversion", key="stop_vc"), sg.Radio("Input listening", "function", key="im", default=False, enable_events=True), sg.Radio("Voice Conversion", "function", key="vc", default=True, enable_events=True), sg.Text("Algorithm delay (ms):"), sg.Text("0", key="delay_time"), sg.Text("Inference time (ms):"), sg.Text("0", key="infer_time")]
            ]
            self.window = sg.Window("Seed-VC - GUI", layout=layout, finalize=True)
            self.event_handler()

        def event_handler(self):
            global flag_vc
            while True:
                event, values = self.window.read()
                if event == sg.WINDOW_CLOSED:
                    self.stop_stream()
                    exit()
                if event == "reload_devices" or event == "sg_hostapi":
                    self.gui_config.sg_hostapi = values["sg_hostapi"]
                    self.update_devices(hostapi_name=values["sg_hostapi"])
                    if self.gui_config.sg_hostapi not in self.hostapis: self.gui_config.sg_hostapi = self.hostapis[0]
                    self.window["sg_hostapi"].Update(values=self.hostapis, value=self.gui_config.sg_hostapi)
                    if (self.gui_config.sg_input_device not in self.input_devices and len(self.input_devices) > 0):
                        self.gui_config.sg_input_device = self.input_devices[0]
                    self.window["sg_input_device"].Update(values=self.input_devices, value=self.gui_config.sg_input_device)
                    if self.gui_config.sg_output_device not in self.output_devices:
                        self.gui_config.sg_output_device = self.output_devices[0]
                    self.window["sg_output_device"].Update(values=self.output_devices, value=self.gui_config.sg_output_device)

                if event == "start_vc" and not flag_vc:
                    if self.set_values(values):
                        printt("cuda_is_available: %s", torch.cuda.is_available())
                        self.start_vc()
                        if self.stream is not None:
                            self.delay_time = self.stream.latency[-1] + values["block_time"] + values["crossfade_length"] + values["extra_time_right"] + 0.01
                            self.window["sr_stream"].update(self.gui_config.samplerate)
                            self.window["delay_time"].update(int(np.round(self.delay_time * 1000)))

                elif event == "diffusion_steps": self.gui_config.diffusion_steps = values["diffusion_steps"]
                elif event == "inference_cfg_rate": self.gui_config.inference_cfg_rate = values["inference_cfg_rate"]
                elif event in ["vc", "im"]: self.function = event
                elif event == "stop_vc": self.stop_stream()
                elif event == "threhold": self.gui_config.threhold = values["threhold"]
                elif event == "mic_gain": self.gui_config.mic_gain = values["mic_gain"]

        def set_values(self, values):
            if not values["reference_audio_path"]: return False
            self.gui_config.reference_audio_path = values["reference_audio_path"]
            self.gui_config.sg_input_device = values["sg_input_device"]
            self.gui_config.sg_output_device = values["sg_output_device"]
            self.gui_config.reference_audio_path = values["reference_audio_path"]
            self.gui_config.sr_type = ["sr_model", "sr_device"][[values["sr_model"], values["sr_device"]].index(True)]
            self.gui_config.diffusion_steps = values["diffusion_steps"]
            self.gui_config.inference_cfg_rate = values["inference_cfg_rate"]
            self.gui_config.max_prompt_length = values["max_prompt_length"]
            self.gui_config.block_time = values["block_time"]
            self.gui_config.crossfade_time = values["crossfade_length"]
            self.gui_config.extra_time_ce = values["extra_time_ce"]
            self.gui_config.extra_time = values["extra_time"]
            self.gui_config.extra_time_right = values["extra_time_right"]
            self.gui_config.threhold = values["threhold"]
            self.gui_config.mic_gain = values["mic_gain"]
            try:
                self.set_devices(values["sg_input_device"], values["sg_output_device"])
            except: pass
            return True

        def start_vc(self):
            if device.type == "cuda": torch.cuda.empty_cache()
            sr_model = self.model_set[-1]["sampling_rate"]
            self.reference_wav, _ = librosa.load(self.gui_config.reference_audio_path, sr=sr_model)
            
            # [修正] 嚴格判定取樣率
            device_sr = self.get_device_samplerate()
            if self.gui_config.sr_type == "sr_model":
                self.gui_config.samplerate = sr_model
            else:
                self.gui_config.samplerate = device_sr
                
            self.gui_config.channels = self.get_device_channels()
            
            self.zc = self.gui_config.samplerate // 50
            self.block_frame = int(np.round(self.gui_config.block_time * self.gui_config.samplerate / self.zc)) * self.zc
            self.block_frame_16k = 320 * self.block_frame // self.zc
            self.crossfade_frame = int(np.round(self.gui_config.crossfade_time * self.gui_config.samplerate / self.zc)) * self.zc
            self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
            self.sola_search_frame = self.zc
            self.extra_frame = int(np.round(self.gui_config.extra_time_ce * self.gui_config.samplerate / self.zc)) * self.zc
            self.extra_frame_right = int(np.round(self.gui_config.extra_time_right * self.gui_config.samplerate / self.zc)) * self.zc
            
            # [修復] 補回缺失變數
            self.skip_head = self.extra_frame // self.zc
            self.skip_tail = self.extra_frame_right // self.zc
            self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc

            self.input_wav = torch.zeros(self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame + self.extra_frame_right, device=self.config.device)
            self.input_wav_res = torch.zeros(320 * self.input_wav.shape[0] // self.zc, device=self.config.device)
            self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.config.device)
            self.fade_in_window = torch.sin(0.5 * np.pi * torch.linspace(0, 1, self.sola_buffer_frame, device=self.config.device))**2
            self.fade_out_window = 1 - self.fade_in_window
            
            self.resampler = tat.Resample(self.gui_config.samplerate, 16000).to(self.config.device)
            if self.gui_config.samplerate != sr_model:
                self.resampler2 = tat.Resample(sr_model, self.gui_config.samplerate).to(self.config.device)
            else:
                self.resampler2 = None
                
            self.vad_speech_detected = False
            self.set_speech_detected_false_at_end_flag = False
            self.start_stream()

        def start_stream(self):
            global flag_vc
            if not flag_vc:
                flag_vc = True
                extra_settings = sd.WasapiSettings(exclusive=True) if "WASAPI" in self.gui_config.sg_hostapi and self.gui_config.sg_wasapi_exclusive else None
                self.stream = sd.Stream(callback=self.audio_callback, blocksize=self.block_frame, samplerate=self.gui_config.samplerate, channels=self.gui_config.channels, dtype="float32", extra_settings=extra_settings)
                self.stream.start()

        def stop_stream(self):
            global flag_vc
            if flag_vc:
                flag_vc = False
                if self.stream:
                    self.stream.abort()
                    self.stream.close()
                    self.stream = None

        def audio_callback(self, indata, outdata, frames, times, status):
            global flag_vc
            start_time = time.perf_counter()
            # [核心優化] Numpy 單聲道轉換 (極速)
            indata_mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()
            
            # [新增] 應用麥克風增益
            indata_mono = indata_mono * self.gui_config.mic_gain
            
            # [核心優化] 能量 VAD (RMS, 0ms latency)
            rms = np.sqrt(np.mean(indata_mono**2))
            db = 20 * np.log10(rms + 1e-9)
            self.vad_speech_detected = db > self.gui_config.threhold
            print(f"VAD: {self.vad_speech_detected} | dB: {int(db)}", flush=True) # 監控日誌
            
            # GPU Buffer Update
            in_tensor = torch.from_numpy(indata_mono).to(device).float()
            self.input_wav = torch.roll(self.input_wav, -self.block_frame)
            self.input_wav[-len(indata_mono):] = in_tensor
            self.input_wav_res = torch.roll(self.input_wav_res, -self.block_frame_16k)
            
            with torch.no_grad():
                res_chunk = self.resampler(self.input_wav[-len(indata_mono)-2000:])
                # 安全賦值
                L = min(len(self.input_wav_res[-self.block_frame_16k:]), len(res_chunk[-self.block_frame_16k:]))
                self.input_wav_res[-L:] = res_chunk[-L:]

            if self.function == "vc":
                infer_wav = custom_infer(
                    self.model_set, self.reference_wav, self.gui_config.reference_audio_path,
                    self.input_wav_res, self.block_frame_16k, self.skip_head, self.skip_tail,
                    self.return_length, int(self.gui_config.diffusion_steps),
                    self.gui_config.inference_cfg_rate, self.gui_config.max_prompt_length,
                    self.gui_config.extra_time_ce - self.gui_config.extra_time,
                )
                if self.resampler2 is not None:
                    infer_wav = self.resampler2(infer_wav)
                    
                if not self.vad_speech_detected:
                    infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])
            else:
                infer_wav = self.input_wav[self.extra_frame :].clone()

            # SOLA (GPU Float32)
            with torch.no_grad():
                req_len = self.sola_buffer_frame + self.sola_search_frame
                if infer_wav.shape[0] < req_len: infer_wav = F.pad(infer_wav, (0, req_len - infer_wav.shape[0]))
                
                conv_input = infer_wav[None, None, : req_len].float()
                cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
                cor_den = torch.sqrt(F.conv1d(conv_input**2, torch.ones(1, 1, self.sola_buffer_frame, device=device)) + 1e-8)
                sola_offset = max(0, min(int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item()), self.sola_search_frame))

                processed_wav = infer_wav[sola_offset:]
                processed_wav[:self.sola_buffer_frame] = processed_wav[:self.sola_buffer_frame] * self.fade_in_window + self.sola_buffer * self.fade_out_window
                self.sola_buffer[:] = processed_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
            
            outdata[:] = processed_wav[:self.block_frame].cpu().numpy().reshape(-1, 1).repeat(self.gui_config.channels, 1)
            
            if flag_vc:
                self.window["infer_time"].update(int((time.perf_counter() - start_time) * 1000))

        def update_devices(self, hostapi_name=None):
            global flag_vc
            flag_vc = False
            sd._terminate(); sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for h in hostapis:
                for d_idx in h["devices"]: devices[d_idx]["hostapi_name"] = h["name"]
            self.hostapis = [h["name"] for h in hostapis]
            if hostapi_name not in self.hostapis: hostapi_name = self.hostapis[0]
            self.input_devices = [d["name"] for d in devices if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name]
            self.output_devices = [d["name"] for d in devices if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name]
            self.input_devices_indices = [d["index"] if "index" in d else d["name"] for d in devices if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name]
            self.output_devices_indices = [d["index"] if "index" in d else d["name"] for d in devices if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name]

        def set_devices(self, input_device, output_device):
            try:
                sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
                sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]
            except: pass
            printt("Input: %s, Output: %s", input_device, output_device)

        def get_device_samplerate(self):
            return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])

        def get_device_channels(self):
            d_in = sd.query_devices(device=sd.default.device[0])
            d_out = sd.query_devices(device=sd.default.device[1])
            return min(d_in["max_input_channels"], d_out["max_output_channels"], 2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--fp16", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda"
    device = torch.device(cuda_target if torch.cuda.is_available() else "cpu")
    gui = GUI(args)
