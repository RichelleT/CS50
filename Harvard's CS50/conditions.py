from cs50 import get_int 

#gets number or int from user/keyboard
x = get_int("x: ")

#gets number/int from user/keyboard
y = get_int("y: ")

if x < y:
	print("x is less than y")
elif x > y:
	print("x is greater than y")
else:
	print("x is equal to y")