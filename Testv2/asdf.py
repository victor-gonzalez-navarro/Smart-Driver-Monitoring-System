import serial
import time

ser = serial.Serial('/dev/cu.usbmodem1421', 9600)
Sentado = 0
time.sleep(2)
while(True):

    ser.reset_input_buffer()
    time.sleep(1)

    #while(not ser.inWaiting()>0):
     #   ()

    #ser.reset_input_buffer()
    Seat = ser.readline()  #// Read the PulseSensor's value. // Assign this value to the "Signal" variable.
    #Seat = int(Seat)
    print(Seat)
    caca = ser.readline()
    #caca = int(caca)
    #caca = ser.in_waiting
    #print(caca)
    # print(caca)
