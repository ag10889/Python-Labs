
print("Hello world")


A = 'Ace'
K = 10
Q = 10
J = 10




Hearts = [A, 2, 3, 4, 5, 6, 7, 8, 9, K, Q, J]
Spades = [A, 2, 3, 4, 5, 6, 7, 8, 9, K, Q, J]
Clubs = [A, 2, 3, 4, 5, 6, 7, 8, 9, K, Q, J]
Diamonds =[A, 2, 3, 4, 5, 6, 7, 8, 9, K, Q, J]
    
for i in range(len(Hearts)):
    Hearts.append(i + 'S')
print(Hearts)