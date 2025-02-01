def BinaryToDecimal(BinaryNumber):
        decimal = 0
        i = 0

        while (BinaryNumber != 0):
            position = BinaryNumber % 10
            decimal = decimal + position * (2**i)
            BinaryNumber = BinaryNumber //10
            i+= 1
        return decimal
def onesCompliment(BinaryNumber):
    BinaryNumber = abs(BinaryNumber)
    value = ""
    for i in range(len(str(BinaryNumber))):
        if (str(BinaryNumber)[i]) == 0:
            value[i] = 1
        elif (str(BinaryNumber)[i] == 1):
            value[i] = 0
    return value

def twosCompliment(DecimalNumber):
    
    return onesCompliment(DecimalNumber)
    

def DecimalToBinary(DecimalNumber):
    DecimalNumber = abs(DecimalNumber)
    conversionsString = ""
    while (DecimalNumber > 0):
        remainder = DecimalNumber%2
        conversionsString = str(remainder) + conversionsString
        DecimalNumber = DecimalNumber // 2
    return conversionsString
    

print("Enter a 1 to convert Decimal to Binary \
            Enter a 2 to convert Binary to Decimal \
          Enter a 0 to end the program ")
userinput = input("Enter the number")
if (userinput == "1"):
    conversionNumber = int(input("Enter the Decimal number to be converted "))
    if (conversionNumber < 0):
        print(twosCompliment(conversionNumber))
    else:
        print(DecimalToBinary(conversionNumber))
       
elif(userinput == "2"):
    conversionNumber = int(input("Enter the Binary number to be converted "))
    print(BinaryToDecimal(conversionNumber))
    
else:
    print("Please enter valid input")
