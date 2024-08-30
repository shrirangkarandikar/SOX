
def fractionAddition(expression):
    import math
    i=0
    numerator=[]
    denominator=[]
    while(i<len(expression)):
        if(expression[i]=="+"):
            numerator.append(int(expression[i+1]))
            denominator.append(int(expression[i+3]))
            i+=4
        elif(expression[i]=="-"):
            numerator.append((-1)*int(expression[i+1]))
            denominator.append(int(expression[i+3]))
            i+=4
    if(all(map(lambda x: x == denominator[0], denominator))):
        fraction_num=sum(numerator)
        fraction_den=denominator[0]

    else:
        lcd=0
        def lcm(a, b):
            return abs(a * b) // math.gcd(a, b)
        for denom in denominator[0:]:
            lcd = lcm(lcd, denom)

        mult=[lcd//denominator for denominator in range(len(denominator))]
        fraction_num=numerator*mult
        fraction_den=lcd


        print(fraction_num,"/",fraction_den)

fractionAddition("-9/3+6/5-8/1")