import PIL.ImageGrab as img_grb
from PIL import Image

import numpy as np

import tensorflow as tf

#import time

class GetInput():
	def __init__(self, difficulty):
		self.model = tf.keras.models.load_model("minesweeper_tile_recognition_2.model")
	
		self.difficulty = difficulty
		
		self.tile_values = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: -1, 10: 10, 11: -100, 12: 0}
	
		if self.difficulty == "begginer": # 250% zoom
			self.rows = 9
			self.cols = 9
			self.mines = 10
			
			self.crop_box = (75, 281, 436, 642)
			
			self.reset_pos = (257, 218)
			
		elif self.difficulty == "intermediate": # 250% zoom
			self.rows = 16
			self.cols = 16
			self.mines = 40
			
			#self.crop_box = (75, 281, 716, 922) # school comp
			self.crop_box = (75, 317, 716, 958) # home comp
			
			self.reset_pos = (398, 218)
			
		elif self.difficulty == "expert": # 175% zoom
			self.rows = 16
			self.cols = 30
			self.mines = 99
			
			self.crop_box = (53, 220, 894, 669)
			
			self.reset_pos = (474, 177)
			
		else:
			print("Wrong difficulty.")
	
	def prepare_img(self, screen, box):
		tile = screen.crop(box)
		tile = tile.resize((40, 40))
		tile = np.asarray(tile, dtype="int32")
		tile = tile / 255
		
		tile = tile.reshape(-1, 40, 40, 3)
		
		return tile
	
	def read_board(self):
		#start_time = time.time()
	
		screen = img_grb.grab()
		screen = screen.crop(self.crop_box)

		board = np.zeros((self.rows, self.cols))
		
		if self.difficulty == "intermediate":
			for nx in range(0, self.cols):
				for ny in range(0, self.rows):
					box = (1 + 40 * nx, 1 + 40 * ny, 41 + 40 * nx, 41 + 40 * ny)
					
					prediction = self.model.predict([self.prepare_img(screen, box)])
					
					filtered_prediction = np.argmax(prediction[0])
					
					board[ny, nx] = self.tile_values[filtered_prediction]
					
		#print("--- %s seconds ---" % (time.time() - start_time))
												
		return board