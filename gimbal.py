import serial 
import time
import os

ser = serial.Serial('/dev/ttyUSB0',115200)  # open serial port
time.sleep(1)

def speed_control(speed_pitch = 0, speed_roll = 0, speed_yaw = 0):
    angle_pitch = 0
    angle_roll = 0
    angle_yaw = 0
    mode = 1
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
    ser.write(bytes_to_send)
    ser.read(6)

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

def request_datas():
    mode = 281
    command_id = 88 
    payload_size = 10
    header_check_sum = (command_id + payload_size) % 256
    command_id_1u = command_id.to_bytes(1, byteorder="little")
    payload_size_1u = payload_size.to_bytes(1, byteorder="little")
    header_check_sum_1u = header_check_sum.to_bytes(1, byteorder="little")

    payload = mode.to_bytes(4, byteorder="little", signed=True) 
    payload += (0).to_bytes(6, byteorder="little", signed=True) 
    payload_check_sum = 0
    for byte_int in payload:
        payload_check_sum = (payload_check_sum + byte_int) % 256
    payload_check_sum_1u = payload_check_sum.to_bytes(1, byteorder="little")
    
    bytes_to_send = b'\x3e' + command_id_1u + payload_size_1u + header_check_sum_1u \
                + payload + payload_check_sum_1u
    ser.write(bytes_to_send)
    
def get_datas():
    payload = ser.read(31)
    payload = payload[4:30]
    imu_angle = []
    imu_speed = []
    imu_acc = []
    joint_angle = []
    for i in range(12):
        data = int.from_bytes(payload[i*2 + 2: i*2 + 4], byteorder='little', signed=True)
        if i in [0,1,2]:
            imu_angle.append(data * 0.02197265625)
        if i in [3,4,5]:
            joint_angle.append(data * 0.02197265625)
        if i in [6,7,8]:
            imu_speed.append(data * 0.06103701895)
        if i in [9,10,11]:
            imu_acc.append(data / 512)
       # print("%15.3f"%data)
#    print('%15.2f   %15.2f   %15.2f'%(imu_angle[0], imu_angle[1], imu_angle[2]),end = '\r')
#    print('%15.2f   %15.2f   %15.2f'%(joint_angle[0], joint_angle[1], joint_angle[2]),end = '\r')
#    print('%15.2f   %15.2f   %15.2f'%(imu_speed[0], imu_speed[1], imu_speed[2]),end = '\r')
#    print('%15.2f   %15.2f   %15.2f'%(imu_acc[0], imu_acc[1], imu_acc[2]),end = '\r')
    return imu_angle,imu_speed,imu_acc,joint_angle

if __name__ == '__main__':
    while(1):
        request_datas()
        get_datas()
