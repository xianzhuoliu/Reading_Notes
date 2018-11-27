
# coding: utf-8

# In[ ]:


class Array(object):
    """Represents an array."""

    def __init__(self, capacity, fillValue = None):
        """Capacity is the static size of the array.
        fillValue is placed at each position."""
        self._items = list()
        self._capacity = capacity
        self._fillValue = fillValue
        for count in range(capacity):
            self._items.append(fillValue)
        if fillValue is not None:
            self._logicalSize = capacity
        else:
            self._logicalSize = 0

                
    def size(self):
        return self._logicalSize

    def __len__(self):
        """-> The capacity of the array."""
        return len(self._items)

    def __str__(self):
        """-> The string representation of the array."""
        return str(self._items)

    def __iter__(self):
        """Supports iteration over a view of an array."""
        return iter(self._items)

    def __getitem__(self, index):
        """Subscript operator for access at index."""
        if index < 0 or index >= self.size():
            raise KeyError('index out of bound')
        return self._items[index]

    def __setitem__(self, index, newItem):
        """Subscript operator for replacement at index."""
        if index < 0 or index >= len(self):
            raise KeyError('index out of bound')
        if self._items[index] == None and newItem != None:
            self._logicalSize += 1
        if self._items[index] != None and newItem == None:
            self._logicalSize -= 1
        self._items[index] = newItem
        
        
    def grow(self, n):  # n是增加的单位
        """increase the physical size of array by n units"""
        temp = Array(len(self) + n)
#         print(temp)
        for j in range(self.size()):
            temp[j] = self._items[j]
#         self = temp                           # 那笔内存依然放在那里，你只是改变了self的指向而已
#         print('this is self:', self)
        self._items = temp._items               #这样才能真正改变内存空间
        self._logicalSize = temp.size()
        
    
    def shrink(self, n):
        if len(self) - n < self._capacity:
            raise Exception('out of bound')
        else:
            temp = Array(len(self) - n)
            for i in range(self.size()):
                temp[i] = self._items[i]
            self._items = temp._items
            self._logicalSize = temp.size()
        
    def insert(self, targetIndex, newItem):
        if targetIndex < 0:
            raise Exception('out of bound')
        if targetIndex > self.size():
            self[self._logical] = newItem
        for i in range(self.size(), targetIndex, -1):
            self[i] = self[i - 1]
        self[targetIndex] = newItem
        
    def pop(self, targetIndex):
        if targetIndex < 0 or targetIndex >= self.size():
            raise Exception('out of bound')
        targetItem = self[targetIndex]
        for i in range(targetIndex, self.size() - 1):
            self[i] = self[i + 1]
        self[self.size() - 1] = self._fillValue
        return targetItem
    
    def __eq__(self, other):
        if not isinstance(other, Array):
            return False
        elif other.size() != self.size():
            return False
        elif other._items == self._items:
            return True
        
    def info(self):
        print(self)
        print('length: ', len(self))
        print('logicalsize: ', self.size())

