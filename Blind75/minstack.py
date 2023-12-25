class MinStack:
    def __init__(self):
        self.stack = [] #Creates an empty list
        self.min_stack = [] #Create an emptyn min_stack list to contain min val

    def push(self, val: int) -> None:
        self.stack.append(val) #pushes (appends) val to the head node
        current_min = min(val, self.min_stack[-1] if self.min_stack else val) #checks if the value is empty before comparing if the current value needs to be updated.
        self.min_stack.append(current_min) #pushes the new current min

    def pop(self) -> None:
        if self.stack: #checks if filled
            self.stack.pop()
            self.min_stack.pop()

    def top(self) -> int:
        if self.stack: #checks if filled
            return self.stack[-1]

    def getMin(self) -> int:
        if self.min_stack: #checks if filled
            return self.min_stack[-1]
