import random 

binary_arr = ''
for i in range(10):
    binary_arr+=random.choice(['0','5'])

print(binary_arr)
new_arr = ""
for i in binary_arr:
    if int(i) == 5:
        new_arr+="3"
    else:
        new_arr+="0"
print(new_arr)
