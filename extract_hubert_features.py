import argparse
import os
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, HubertModel

from utils import load_filepaths_and_text


def load_wav_16k(path):
  wav, _ = librosa.load(path, sr=16000, mono=True)
  wav = torch.from_numpy(wav).float()
  if wav.abs().max() > 0:
    wav = wav / wav.abs().max()
  return wav


def extract_feature(model, extractor, wav_tensor, device):
  inputs = extractor(
    wav_tensor.numpy(),
    sampling_rate=16000,
    return_tensors="pt",
    padding=False
  )
  inputs = {k: v.to(device) for k, v in inputs.items()}
  with torch.no_grad():
    outputs = model(**inputs)
  feat = outputs.last_hidden_state.squeeze(0).transpose(0, 1).cpu()
  return feat


def build_output_path(wav_path, suffix, out_dir=None):
  wav_path = Path(wav_path)
  if out_dir is None:
    return wav_path.with_suffix(suffix)
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  return out_dir / f"{wav_path.stem}{suffix}"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--filelists", nargs="+", required=True)
  parser.add_argument("--model-name", default="facebook/hubert-base-ls960")
  parser.add_argument("--suffix", default=".ssl.pt")
  parser.add_argument("--out-dir", default=None)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args()

  extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
  model = HubertModel.from_pretrained(args.model_name).to(args.device)
  model.eval()
  for p in model.parameters():
    p.requires_grad = False

  seen = set()
  for filelist in args.filelists:
    for row in load_filepaths_and_text(filelist):
      wav_path = row[0]
      if wav_path in seen:
        continue
      seen.add(wav_path)
      if not os.path.isfile(wav_path):
        print(f"skip missing {wav_path}")
        continue
      out_path = build_output_path(wav_path, args.suffix, args.out_dir)
      if out_path.exists():
        continue

      wav = load_wav_16k(wav_path)
      feat = extract_feature(model, extractor, wav.to("cpu"), args.device)
      torch.save(feat, out_path)
      print(f"saved {out_path}")


if __name__ == "__main__":
  main()
