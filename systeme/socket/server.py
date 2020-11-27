import socket as sc

address = ("", 7001)


with sc.socket(sc.AF_INET, sc.SOCK_STREAM) as socket:
    socket.bind(address)
    socket.listen()
    connection, address = socket.accept()
    with connection:
        print('connected to server', address)
        while True:
            data = connection.recv(1024)
            connection.sendall(data)
            if not data: 
                break
            print(data)
