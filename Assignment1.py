import random

# get user input
length = int(input("Enter the length of the list: "))
min_val = int(input("Enter the minimum value: "))
max_val = int(input("Enter the maximum value: "))
operator = input("Enter operator (+, -, *, /): ")
sort_order = input("Enter sort order (asc or desc): ")

# generate the list of random numbers
random_list = [random.randint(min_val, max_val) for i in range(length)]

# perform arithmetic operation on the list
if operator == '+':
    result_list = [num + 1 for num in random_list]
elif operator == '-':
    result_list = [num - 1 for num in random_list]
elif operator == '*':
    result_list = [num * 2 for num in random_list]
elif operator == '/':
    result_list = [num / 2 for num in random_list]
else:
    print("Invalid operator")

# sort the result list
if sort_order == 'asc':
    sorted_list = sorted(result_list)
elif sort_order == 'desc':
    sorted_list = sorted(result_list, reverse=True)
else:
    print("Invalid sort order")

# print the result
print("Random list:", random_list)
print("Result list:", result_list)
print("Sorted list:", sorted_list)
