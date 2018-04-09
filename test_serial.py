# -*- coding: utf-8 -*-  
import serial
import time

ser=serial.Serial("COM3",115200,timeout=1)

i=1
while 1:
    i+=1
    test = '001002003004'
    test='@%03d%03d%03d%03d@'%(i%1000,(1+i)%1000,(3+i)%1000,(i+4)%1000)
    #test="from client"
    ser.write("AT+CIPSEND=0,"+str(len(test))+"\r\n")
    # response = ser.read(100)
    # print(response)
    time.sleep(1)

    ser.write(test)#+"\r\n")
    res = ser.read(100)
    print 'GET----------------'
    flag = '+IPD'
    if flag in res:
        print res[res.index(flag)+len(flag)+1:].split('\n')[0]
    else:
        print res
    # response = ser.read(100)
    # print (response)
    #print test

    time.sleep(1)

# if(ser.is_open):
#     ser.close()
