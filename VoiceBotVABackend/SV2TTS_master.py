import sys
sys.path.append("../SV2TTS")
from synthesizer.inference import Synthesizer
from synthesizer.preprocess import create_embeddings
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import librosa
import soundfile as sf


import numpy as np


SAMPLE_RATE = 20000
class ModelMaster():

  def __init__(self):
    saved_modal_base_path = Path("../SV2TTS/saved_models/default/")
    #note: the model is not loaded directy from the file, the wights are stored then the 
    encoder.load_model(Path("../SV2TTS/saved_models/default/encoder.pt"))
    self.synthesizer = Synthesizer(Path("../SV2TTS/saved_models/default/synthesizer.pt"))
    vocoder.load_model( Path("../SV2TTS/saved_models/default/vocoder.pt"))
  
  def _upload_audio(self,fpath):
    self.wav, source_sr = librosa.load(str(fpath), sr=None)

  def _compute_embedding(self):
    self.embedding = None
    self.embedding = encoder.embed_utterance(encoder.preprocess_wav(self.wav, SAMPLE_RATE))

  def synthesize(self, text):
    specs = self.synthesizer.synthesize_spectrograms([text], [self.embedding])
    generated_wav = vocoder.infer_waveform(specs[0])
    self.generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")

  def _save_to_file(self,fpath):
    sf.write(fpath, self.generated_wav, SAMPLE_RATE)

if __name__ == '__main__':
  x = ModelMaster()
  x._upload_audio(Path("../SV2TTS/samples/p240_00000.mp3"))
  x._compute_embedding()
  x.synthesize("I can now make this say whatever I want, in almost real time alksjdhflakjsdhflkajsdh")
  x._save_to_file("./test.wav")


