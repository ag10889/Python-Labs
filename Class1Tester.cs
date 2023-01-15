using System;
namespace lab1C
{
    public static void Main(string[] args)
    {
        char[,] carArray = Class1.make_forward();
        for (int row = 0;row<4;row++)
        {
            for (int column = 0; column < carArray[row].Length; column++)
            {
             Console.Write(carArray[row, column]);
            }
        }
    }
}

