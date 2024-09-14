from flask import Flask, request, jsonify
import serial
import time

app = Flask(__name__)

# Set up the serial communication with Arduino
try:
    arduino = serial.Serial('COM3', 9600, timeout=1)  # Use the correct COM port for your Arduino
    time.sleep(2)  # Wait for the connection to establish
except Exception as e:
    print(f"Error connecting to Arduino: {e}")

def send_word_made():
    if arduino.is_open:
        arduino.write(b'wordMade')  # Send the wordMade command to Arduino
        return "Sent 'wordMade' to Arduino"
    return "Arduino connection not open"

def reset_word_made():
    if arduino.is_open:
        arduino.write(b'resetWordMade')  # Send the resetWordMade command
        return "Sent 'resetWordMade' to Arduino"
    return "Arduino connection not open"

@app.route('/send_command', methods=['POST'])
def handle_command():
    data = request.json
    command = data.get('command')
    if command == 'wordMade':
        result = send_word_made()
    elif command == 'resetWordMade':
        result = reset_word_made()
    else:
        result = "Unknown command"
    
    return result

print(send_word_made())