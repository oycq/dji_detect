import serial 
import time
import os

ser = serial.Serial('/dev/ttyUSB0',115200)  # open serial port
time.sleep(1)

def angle_control(angle_pitch = 0, angle_roll = 0, angle_yaw = 0,
            speed_pitch = 0, speed_roll = 0, speed_yaw = 0,
            mode = 1):
    command_id = 67
    payload_size = 3 + 2 * 3 + 2 * 3
    header_check_sum = (command_id + payload_size) % 256
    command_id_1u = command_id.to_bytes(1, byteorder="little")
    payload_size_1u = payload_size.to_bytes(1, byteorder="little")
    header_check_sum_1u = header_check_sum.to_bytes(1, byteorder="little")

    mode_1u = (mode).to_bytes(1, byteorder="little")

    speed_roll_2s  = (int(speed_roll/0.1220740379)).to_bytes(2, byteorder="little", signed=True)
    angle_roll_2s  = (int(angle_roll/0.02197265625)).to_bytes(2, byteorder="little", signed=True)
    speed_pitch_2s = (int(speed_pitch/0.1220740379)).to_bytes(2, byteorder="little", signed=True)
    angle_pitch_2s = (int(angle_pitch/0.02197265625)).to_bytes(2, byteorder="little", signed=True)
    speed_yaw_2s   = (int(speed_yaw/0.1220740379)).to_bytes(2, byteorder="little", signed=True)
    angle_yaw_2s   = (int(angle_yaw/0.02197265625)).to_bytes(2, byteorder="little", signed=True)
    
    payload = mode_1u + mode_1u +mode_1u \
                + speed_roll_2s + angle_roll_2s \
                + speed_pitch_2s + angle_pitch_2s \
                + speed_yaw_2s + angle_yaw_2s
    payload_check_sum = 0
    for byte_int in payload:
        payload_check_sum = (payload_check_sum + byte_int) % 256
    payload_check_sum_1u = payload_check_sum.to_bytes(1, byteorder="little")
    
    bytes_to_send = b'\x3e' + command_id_1u + payload_size_1u + header_check_sum_1u \
                + payload + payload_check_sum_1u
    o = time.time() *1000
#    for item in bytes_to_send:
#        print(item)
    ser.write(bytes_to_send)
    o1 = time.time() *1000
    #ser.flush()
    o2 = time.time() *1000
    
#    o = time.time() *1000
#    s = "a" * 64 
#    print(len(s))
#    ser.write(s.encode('ASCII'))
#    o1 = time.time() *1000
#    ser.flush()
#    o2 = time.time() *1000
#    print("1    %7.4f"%(o1-o))
#    print("2    %7.4f"%(o2-o1))
#    asd()
    a = ser.read(6)


    o3 = time.time() *1000
#    a = ser.read(7)
    o4 = time.time() *1000
#    os.system('clear')
#    print(len(bytes_to_send))
#    print("1    %7.4f"%(o1-o))
#    print("2    %7.4f"%(o2-o1))
#    print("3    %7.4f"%(o3-o2))
#    print("4    %7.4f"%(o4-o3))
#    print("4    %7.4f"%(o4-o))
    return o3-o ,o4-o3


def send_cmd(command_id):
    payload_size = 0
    header_check_sum = (command_id + payload_size) % 256
    command_id_1u = command_id.to_bytes(1, byteorder="little")
    payload_size_1u = payload_size.to_bytes(1, byteorder="little")
    header_check_sum_1u = header_check_sum.to_bytes(1, byteorder="little")
    payload = (0).to_bytes(1, byteorder="little")
    payload_check_sum_1u = (0).to_bytes(1, byteorder="little")
    bytes_to_send = b'\x3e' + command_id_1u + payload_size_1u + header_check_sum_1u\
                + payload + payload_check_sum_1u
    ser.write(bytes_to_send)
    ser.read(6)



def motor_off():
    send_cmd(109)

def motor_on():
    send_cmd(77)

#motor_off()
#motor_on()

#while(1):
#    angle_control(angle_pitch = 0, angle_roll = 0, angle_yaw = 0)
#    angle_control(angle_pitch = -90, angle_roll = 0, angle_yaw = 0)
#for i in range(10):
#    for j in range(100):
#        angle = (50-j)*0.4*(1-i%2*2)
#        angle_control(angle, angle, angle)
#        time.sleep(0.01)

