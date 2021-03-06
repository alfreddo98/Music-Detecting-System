"""
Author: Alfredo Sánchez Sánchez
Date: 25/06/2020
Project name: Music Detecting System (TFG)
Description: Final Degree Proyect Final project for the Telecomunication degree in ICAI Comillas, in Madrid (Spain) It consists on a system installed in raspbian using Unicorn Hat a 8x8 WS2812B RGB leds, the leds will shine at the same rithm and compass as the music that is sounding and will also detect the genre of the music that is being sounded. 
"""
# Import libraries
import pyaudio
import wave
form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 4096 # 2^12 samples for buffer
record_secs = 20 # seconds to record
#dev_index = 2 # where the device is (in index) (use p.get_device_info_by_index)
wav_output_filename = 'audio.wav' # name of .wav file
audio = pyaudio.PyAudio() # create pyaudio instantiation

# create pyaudio stream
stream = audio.open(format = form_1,rate = samp_rate,channels = chans, 
                    input = True, 
                    frames_per_buffer=chunk)
print("recording")
frames = []

# loop through stream and append audio chunks to frame array
for ii in range(0,int((samp_rate/chunk)*record_secs)):
    data = stream.read(chunk)
    frames.append(data)

print("finished recording")

# stop the stream, close it, and terminate the pyaudio instantiation
stream.stop_stream()
stream.close()
audio.terminate()

# save the audio frames as .wav file
wavefile = wave.open(wav_output_filename,'wb')
wavefile.setnchannels(chans)
wavefile.setsampwidth(audio.get_sample_size(form_1))
wavefile.setframerate(samp_rate)
wavefile.writeframes(b''.join(frames))
wavefile.close()
