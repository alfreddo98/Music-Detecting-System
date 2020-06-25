# Import libraries, specially the library recorder qhere we have the methods we will use to record the song repetitively
from recorder import *
from time import clock
import sys
import numpy
import keyboard
bpm_list = []
prev_beat = clock()
low_freq_avg_list = []
    """
@description: Method that will detect beats repetitively, it will make the leds turn on and off too
Inputs: A variable to know whether the leds are on or off, the bpm_average computed adn the genre detected
Outputs: Nothing just the program with the leds turning on and off with the corresponding figure
"""    
def detect_beats():
	if not input_recorder.has_new_audio: 
		return

	# get x and y values from FFT
	xs, ys = input_recorder.fft()
    
	# calculate average for all frequency ranges
	y_avg = numpy.mean(ys)

	# calculate low frequency average
	low_freq = [ys[i] for i in range(len(xs)) if xs[i] < 1000]
	low_freq_avg = numpy.mean(low_freq)
    
	global low_freq_avg_list
	low_freq_avg_list.append(low_freq_avg)
	cumulative_avg = numpy.mean(low_freq_avg_list)
    
	bass = low_freq[:int(len(low_freq)/2)]
	bass_avg = numpy.mean(bass)
	#print("bass: {:.2f} vs cumulative: {:.2f}".format(bass_avg, cumulative_avg))
    
	# check if there is a beat
	# song is pretty uniform across all frequencies
	if (y_avg > 10 and (bass_avg > cumulative_avg * 1.5 or(low_freq_avg < y_avg * 1.2 and bass_avg > cumulative_avg))):
		global prev_beat
		curr_time = clock()

		if curr_time - prev_beat > 60/180: # 180 BPM max
			#print("beat")
			print(curr_time - prev_beat)
			global bpm_list
			bpm = int(60 / (curr_time - prev_beat))
			print(bpm)
			if len(bpm_list) < 4:
				if bpm > 60:
					bpm_list.append(bpm)
			else:
				bpm_avg = int(numpy.mean(bpm_list))
				print(bpm_avg)
				if abs(bpm_avg - bpm) < 35:
					bpm_list.append(bpm)
			# reset the timer
			prev_beat = curr_time
        # shorten the cumulative list to account for changes in dynamics
	if len(low_freq_avg_list) > 50:
		low_freq_avg_list = low_freq_avg_list[25:]
		print("REFRESH!!")

	# keep two 8-counts of BPMs so we can maybe catch tempo changes
	if len(bpm_list) > 24:
		bpm_list = bpm_list[8:]

	# reset song data if the song has stopped
	if y_avg < 10:
		bpm_list = []
		low_freq_avg_list = []
		print("new song")

if __name__ == "__main__":

	input_recorder = InputRecorder()
	input_recorder.start()
	while True:
		detect_beats()
	# clean up
		if (keyboard.is_pressed('p')):
			print("Se presiono, paramos")
			input_recorder.close()
			sys.exit()
			break
		




