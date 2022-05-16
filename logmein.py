v8 = ":\"AL_RT^L*.?+6/46"
v7 = hex(28537194573619560)[2:]#去掉0x
v7 = str(bytes.fromhex(v7))[-2:1:-1]
v6 = 7
flag = ""
for i in range(0,len(v8)):
    flag+=chr(ord(v7[i%v6])^ord(v8[i]))
print(flag)