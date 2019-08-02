"""
Ask the user for a number.
Depending on whether the number is even or odd,
print out an appropriate message to the user.
"""

num = input("Enter any number: ")

mod = int(num) % 2

if mod > 0:
    print("You have entered an odd number.")
else:
    print("You have entered an even number.")
