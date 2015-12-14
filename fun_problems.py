

def reverse_string(str):

    string = list(str)
    
    n=0
    while(string[n] != '\0'):
        n += 1
    
    #n = len(str)
    for i in range(n/2):
        temp = str[i]
        string[i] = str[n-1-i]
        string[n-1-i] = temp

    return "".join(string)

#string = '3jbsdf9sdaf32r2938y4bskjasbdg9823r923824fbwaed9f8\0'
#print string[-1]
#print reverse_string(string)
##--------------------------------------------------------------------

def string_to_ascii(str):
    n = len(str)
    counts = [0] * 256
    for i in range(n):
        counts[ord(str[i])] += 1

    return counts

def check_string_permutation(str1, str2):
    string1 = list(str1)
    string2 = list(str2)

    ascii1 = string_to_ascii(string1)
    ascii2 = string_to_ascii(string2)

    same = False
    if ascii1 == ascii2:
        same = True

    return same


str1 = 'asdasdasd'
str2 = 'dasasdsad'
str3 = 'asdafadas'

#print check_string_permutation(str1, str2)
#print check_string_permutation(str3, str2)
#print check_string_permutation(str1, str3)

    
#------------------------------------------------

#create a linked list class

class Node(object):
    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next_node = next_node

    def get_data(self):
        return self.data

    def get_next(self):
        return self.next_node

    def set_next(self, new_next):
        self.next_node = new_next

class LinkedList(object):
    def __init__(self, head=None):
        self.head = head

    def insert(self, data):
        new_node = Node(data)
        new_node.set_next(self.head)
        self.head = new_node

    def size(self):
        current = self.head
        count = 0
        while current:
            count += 1
            current = current.get_next()
        return count
        
    def search(self, data):
        current = self.head
        found = False
        while current and not found:
            if current.get_data() == data:
                found = True
            else:
                current = current.get_next()
                if current is None:
                    raise ValueError('data not in list')
        return current


    def delete(self, data):
        current = self.head
        previous = None
        found = False
        while current and not found:
            if current.get_data() == data:
                found = True
            else:
                previous = current
                current = current.get_next()
        if current is None:
            return ValueError('data not in list')
        if previous is None:
            self.head = current.get_next()
        else:
            previous.set_next(current.get_next())

        
T = LinkedList()

T.insert(1)
T.insert('hi')
T.insert(128)
T.insert('mags')
T.insert('keys')
T.insert(204)
T.insert('qe')


# find k from last element of linked list

def k_from_end(list, k):
    current1 = list.head
    count = 0;
    
    while current1:
        count += 1
        current1 = current1.get_next()
        if count==k:
            current2 = list.head
        if count>k:
            current2 = current2.get_next()
        
    return current2.get_data()

#print k_from_end(T, 6)

#-------------------------------------------------

# delete a node in the middle of a list, only have access to that

def delete(node):
    next = node.get_next()
    node.data = next.get_data()
    node.next_node = next.next_node()

#-------------------------------------------------

# reorder list so all values below some value are first then, rest 

def reorder_list(list, x):
    current = list.head
    list1 = LinkedList()
    list2 = LinkedList()

    while current:
        val = current.get_data()
        if val < x:
            list1.insert(val)
        else:
            list2.insert(val)
        current = current.get_next()

    current = list1.head
    if current is None:
        list1.head = list2.head
    else:
        while current:
            prev = current
            current = current.get_next()
        prev.set_next(list2.head)

    return list1

T = LinkedList()
T.insert(31)
T.insert(123)
T.insert(14)
T.insert(27)
T.insert(3)
T.insert(104)
T.insert(75)

#newlist = reorder_list(T, 40)

#------------------------------------------------------------

#doesn't work because of LIFO
def sum_list_numbers(list1, list2):
    c1 = list1.head
    c2 = list2.head
    sumlist = LinkedList()
    carry = 0
    while c1 or c2 or carry>0:
        if c1 and c2:
            sumlist.insert(carry + c1.get_data() + c2.get_data() % 10)
            if carry + c1.get_data() + c2.get_data() > 9:
                carry = 1
            else:
                carry = 0
            c1 = c1.get_next()
            c2 = c2.get_next()
        elif c1:
            sumlist.insert(carry + c1.get_data() % 10)
            if carry + c1.get_data() > 9:
                carry = 1
            else:
                carry = 0
            c1 = c1.get_next()
        elif c2:
            sumlist.insert(carry + c2.get_data() % 10)
            if carry + c2.get_data() > 9:
                carry = 1
            else:
                carry = 0
            c2 = c2.get_next()
        else:
            sumlist.insert(1)

    return sumlist


T = LinkedList()
T.insert(3)
T.insert(2)
T.insert(9)
T.insert(7)
T.insert(5)
T.insert(6)
T.insert(8)

U = LinkedList()
U.insert(3)
U.insert(2)
U.insert(9)
U.insert(7)
U.insert(5)
U.insert(8)

    
#temp = sum_list_numbers(T, U)
#c = temp.head
#while c:
#    print c.get_data()
#    c = c.get_next()

# ---------------------------------------------------------------------------------------
    
class Stack(object):
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def isEmpty(self):
        if self.items.size()==0:
            return True
        else:
            return False
        

class Queue(object):
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        self.items.pop()

    def size(self):
        return len(self.items)



# make a queue using stacks

class constructed_queue(object):
    def __init__(self, stack_in, stack_out):
        self.stack_in = stack_in
        self.stack_out = stack_out
        self.items = stack_in.items
        
    def enqueue(self, item):
        self.stack_in.push(item)

    def dequeue(self):
        if self.stack_out.isEmpty():
            while not self.stack_in.isEmpty():
                stack_out.push(stack_in.pop())
        return stack_out.pop()
            
        self.stack_out.items.pop()

    def size(self):
        self.stack1.size()



#----------------

class Ref(object):
    def __init__(self, obj):
        self.obj = obj
    def get(self):
        return self.obj
    def set(self, obj):
        self.obj = obj
   
a = Ref([1, 2, 3])
b = a
print a.get()
print b.get()

b.set(2)
print a.get()
print b.get()


class Node(object):
    def __init__(self, data=None, next=None, prev=None):
        self.data = data
        self.next = next
        self.prev = prev

    def get_value(self):
        return self.data

    def get_next(self):
        return self.next

    def set_next(self, next):
        self.next = next

    def get_prev(self):
        return self.prev

    def set_prev(self, prev):
        self.prev = prev

    
class LinkedList(object):
    def __init__(self, head=None, tail=None):
        self.head = head

    def size(self):
        current = self.head
        c = 0;
        while current:
            c += 1
            current = 
        
    def insert(self, data):
        new_node = Node(data)
        prev = self.head.get_prev()
        
