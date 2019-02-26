"""
File: tokens.py
Tokens for processing expressions.
"""

class Token(object):

    UNKNOWN  = 0        # unknown
    
    INT      = 4        # integer
            
    MINUS    = 5        # minus    operator
    PLUS     = 6        # plus     operator
    MUL      = 7        # multiply operator
    DIV      = 8        # divide   operator
    EXP      = 9
    
    LEFTPAR   = 10
    RIGHPAR   = 11

    FIRST_OP = 5        # first operator code

    def __init__(self, value):
        if type(value) == int:
            self._type = Token.INT
        else:
            self._type = self._makeType(value)
        self._value = value

    def isOperator(self):
        return self._type >= Token.FIRST_OP and self._type < 10

    def __str__(self):
        return str(self._value)
    
    def getType(self):
       return self._type
    
    def getValue(self):
       return self._value

    def _makeType(self, ch):
        if   ch == '*': return Token.MUL
        elif ch == '/': return Token.DIV
        elif ch == '+': return Token.PLUS
        elif ch == '-': return Token.MINUS
        elif ch == '^': return Token.EXP
        elif ch == '(': return Token.LEFTPAR
        elif ch == ')': return Token.RIGHPAR
        else:       return Token.UNKNOWN;
    
    def getPrecedence(self, type1): #self._type代表表达式中的, type1表示栈中的
        if type1 == 9:
            return True
        elif (type1 == 7 or type1 == 8):
            return True
        elif (type1 == 5 and self._type == 6) or (type1 == 6 and self._type == 5) or (type1 == self._type):
            #具有相等优先级
            return True
        elif (type1 == 7 and self._type == 8) or (type1 == 8 and self._type == 7) or (type1 == self._type):
            #具有相等优先级
            return True
        else:
            return False
        
        
    
def main():
    # A simple tester program
    plus = Token("+")
    minus = Token("-")
    mul = Token("*")
    div = Token("/")
    exp = Token("^")
    unknown = Token("#")
    anInt = Token(34)
    print(plus, minus, mul, div, unknown, anInt)

if __name__ == '__main__': 
    main()

