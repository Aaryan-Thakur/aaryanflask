import cv2
import sqlite3
import datetime

# Connect to the database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Create a table to store the images
cursor.execute('''CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                image BLOB, captured_at TEXT)''')

# Capture an image from the camera
camera = cv2.VideoCapture(0)
ret, frame = camera.read()

# Save the image data to the database
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
cursor.execute("INSERT INTO images (image, captured_at) VALUES (?, ?)", (frame.tobytes(), timestamp))
conn.commit()

# Close the database connection and camera
cursor.close()
conn.close()
camera.release()
