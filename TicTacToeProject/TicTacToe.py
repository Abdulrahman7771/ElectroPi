import sys

class Player:
    def __init__(self):
        self.__score = 0
    def SetName(self, name):
        self.__name = name
    def GetName(self):
        return self.__name
    def SetSymbol(self, symbol):
        self.__symbol = symbol
    def IncreaseSecore(self):
        self.__score += 1
    def GetScore(self):
        return self.__score
    def PPlay(self):
        while (True):
            try:
                x,y = map(int, input(f"{self.__name} Please enter the row and column in numbers: ").split())
            except:
                print(f"{self.__name} Please enter the row and column in numbers like this '1 3'")
            else:
                break
        while( (int(x)<1 or int(x)>3) or (int(y) < 1 or int(y) > 3)) :
           x,y = input(f"{self.__name} Please enter valid numbers between 1 and 3: ").split()
           print(x,y) 
        while(not(Board.CheckBoard(int(x)-1,int(y)-1))):
           x,y = input(f"{self.__name} Please enter a free position: ").split() 
           while( (int(x)<1 or int(x)>3) or (int(y) < 1 or int(y) > 3)) :
                x,y = input(f"{self.__name} Please enter valid numbers between 1 and 3: ").split()
                print(x,y) 
        Board.Update(self.__symbol,int(x)-1,int(y)-1)
    def GetSymbol(self):
        return self.__symbol

class Game:
    __board =[["","",""],
              ["","",""],
              ["","",""]]
    P1 = Player()
    P2 = Player()
    winner = ""
    DrawCase = False
    def __init__(self,board=__board,state=""):
        self.__state = state
        self.__board = board
    def Reset(self):
        DrawCase = False
        for i in range(0, len(self.__board)):
            for j in range(0,len(self.__board)):
                self.__board[i][j] = ""
                print("_",end=" ") 
            print("\n") 
    def Update(self,ps,x,y):
        self.__board[x][y]=ps
        for i in range(0, len(self.__board)):
            for j in range(0,len(self.__board)):
                if(self.__board[i][j]==""):
                    print("_",end=" ") 
                else:
                    print(self.__board[i][j],end=" ") 
            print("\n")
    def SetState(self, state):
        self.__state = state
    def GetState(self):
        return self.__state
    def CheckBoard(self,x,y):
        if(self.__board[x][y]==""):
            return True
        else:
            return False
    def EndingCheck(self,symbol):
        for i in range(3):
            if(self.__board[i][0] == symbol and self.__board[i][1]==symbol and self.__board[i][2] == symbol):
                self.SetState("End")
                if(symbol == self.P1.GetSymbol()):
                    self.winner = self.P1.GetName()
                    self.P1.IncreaseSecore()
                else:
                    self.winner = self.P2.GetName()
                    self.P2.IncreaseSecore()
                return
        for j in range(3):
            if(self.__board[0][j]== symbol and self.__board[1][j] == symbol and self.__board[2][j]== symbol):
                self.SetState("End")
                if(symbol == self.P1.GetSymbol()):
                    self.winner = self.P1.GetName()
                    self.P1.IncreaseSecore()
                else:
                    self.winner = self.P2.GetName()
                    self.P2.IncreaseSecore()
                return
        if(self.__board[0][0] == symbol and self.__board[1][1]== symbol and self.__board[2][2]== symbol):
            self.SetState("End")
            if(symbol == self.P1.GetSymbol()):
                    self.winner = self.P1.GetName()
                    self.P1.IncreaseSecore()
            else:
                self.winner = self.P2.GetName()
                self.P2.IncreaseSecore()
            return
        elif (self.__board[0][2]== symbol and self.__board[1][1] == symbol and self.__board[2][0] == symbol):
            self.SetState("End")
            if(symbol == self.P1.GetSymbol()):
                    self.winner = self.P1.GetName()
                    self.P1.IncreaseSecore()
            else:
                self.winner = self.P2.GetName()
                self.P2.IncreaseSecore()
            return
        for i in range(3):
            for j in range(3):
                if(self.__board[i][j]==""):
                    break
                else:
                    if (i == 2 and j == 2 and self.__board[i][j]!=""):
                        self.winner = "None"
                        self.SetState("End")

              
    def EndingSequence(self):
        print(f"{self.P1.GetName()} score = {self.P1.GetScore()} ------- {self.P2.GetName()} score = {self.P2.GetScore()}")
        print(self.winner)
        print("----Won----")
        decision = input("Please enter 'again' to start a new game or 'end' to finish: ")
        while(decision!="again" and decision!="end"):
            decision = input("Please enter 'again' to start a new game or 'end' to finish: ")
        if(decision == 'again'):
            self.Reset()
            self.SetState("P1")
        else:
            print("-------Final Score-------")
            print(f"{self.P1.GetName()} score = {self.P1.GetScore()} ------- {self.P2.GetName()} score = {self.P2.GetScore()}")
            if(self.P1.GetScore()>self.P2.GetScore()):
                print(self.P1.GetName())
                print("-------Winner-------")
            elif(self.P1.GetScore()==self.P2.GetScore()):
                print("-------Draw-------")
            else:
                print(self.P2.GetName())
                print("-------Winner-------")
            input("Please press enter to exit")
            sys.exit()
            
                    

Board = Game()

def Start():
    Board.Reset()
    Board.P1.SetName(input ("Player 1 Please enter your name: "))
    Board.P2.SetName(input ("Player 2 Please enter your name: "))
    Board.P1.SetSymbol (input (f"{Board.P1.GetName()} Please choose Between 'X' and 'O': "))
    while((Board.P1.GetSymbol() != "x" and Board.P1.GetSymbol() != "X") and (Board.P1.GetSymbol() != "o" and Board.P1.GetSymbol() != "O")):
        Board.P1.SetSymbol (input (f"{Board.P1.GetName()} Please choose Between 'X' and 'O': "))
    if(Board.P1.GetSymbol()=="X"or Board.P1.GetSymbol()=="x"):
        Board.P2.SetSymbol("O")
    else:
        Board.P2.SetSymbol ("X")
    Board.SetState("P1")

Start()

while (True):
    if(Board.GetState()=="P1"):
        Board.P1.PPlay()
        Board.EndingCheck(Board.P1.GetSymbol())
        if(Board.GetState()!="End"):
            Board.SetState("P2")
    elif(Board.GetState()=="P2"):
        Board.P2.PPlay()
        Board.EndingCheck(Board.P2.GetSymbol())
        if(Board.GetState()!="End"):
            Board.SetState("P1")
    else:
        Board.EndingSequence()
        

