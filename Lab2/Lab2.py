import random

length = int(input("Enter the length of the list: "))
min_val = int(input("Enter the minimum value: "))
max_val = int(input("Enter the maximum value: "))

# generate the list of random numbers
random_list = [random.randint(min_val, max_val) for i in range(length)]

# sort the list in ascending order
sorted_list = sorted(random_list)

print("Random list:", random_list)
print("Sorted list:", sorted_list)
