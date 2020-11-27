import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 7001        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True:
        msg = input("Enter your message : ")
        if msg == 'stop':
            break
        s.sendall(bytes(msg, 'ascii'))
        data = s.recv(1024)

print('Received', repr(data))