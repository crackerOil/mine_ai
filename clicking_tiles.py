import win32api, win32con
import time

class GetOutput():
	def __init__(self, difficulty):
		self.difficulty = difficulty
		
		if self.difficulty == "beginner": # 250% zoom
			print("Stop.")
			input()
			
		elif self.difficulty == "intermediate": # 250% zoom
			#self.table_coords = (75, 281) # school comp
			self.table_coords = (75, 317) # home comp
			self.reset_coords = (398, 253)
			
		elif self.difficulty == "expert": # 175% zoom
			print("Stop.")
			input()
			
		else:
			print("Wrong difficulty.")
	
	def left_click(self, x, y):
		win32api.SetCursorPos(((self.table_coords[0] + (40 * x) + 20), (self.table_coords[1] + (40 * y) + 20)))
		time.sleep(0.05)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,(self.table_coords[0] + (40 * x) + 20),(self.table_coords[1] + (40 * y) + 20),0,0)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,(self.table_coords[0] + (40 * x) + 20),(self.table_coords[1] + (40 * y) + 20),0,0)
	
	def right_click(self, x, y):
		win32api.SetCursorPos(((self.table_coords[0] + (40 * x) + 20), (self.table_coords[1] + (40 * y) + 20)))
		time.sleep(0.05)
		win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,(self.table_coords[0] + (40 * x) + 20),(self.table_coords[1] + (40 * y) + 20),0,0)
		win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,(self.table_coords[0] + (40 * x) + 20),(self.table_coords[1] + (40 * y) + 20),0,0)
	
	def reset_board(self):
		win32api.SetCursorPos((self.reset_coords[0], self.reset_coords[1]))
		time.sleep(0.05)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,self.reset_coords[0],self.reset_coords[1],0,0)
		win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,self.reset_coords[0],self.reset_coords[1],0,0)