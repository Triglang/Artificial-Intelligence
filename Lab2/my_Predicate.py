from collections import OrderedDict

DEBUG = 1

class Sentences:
    def __init__(self, path):
        self.clauses = []
        with open(path, 'r') as f:
            lines = [line.strip() for line in f]
            
        for line in lines:
            if not line or line.lower() == "kb:" or line.lower() == "query:":
                continue
            if line.startswith('(') and line.endswith(')'):
                line = line[1: -1]
            line = line[:-1]
            literals = []
            
            for lit in line.split("),"):
                literals.append(lit.replace(" ", "") + ")")
            self.clauses.append(tuple(literals))
        
        if DEBUG:
            for item in self.clauses:
                print(item)
            print(self.clauses)
            
    def is_complement(self, literal1, literal2):
        if DEBUG:
            print("literal1 and literal2: ", type(literal1), type(literal2))
        end1 = literal1.find('(')
        end2 = literal2.find('(')
        if literal1.startswith('~') and literal1[1:end1] == literal2[:end2]:
            return True
        elif literal2.startswith('~') and literal1[:end1] == literal2[1:end2]:
            return True
        
        return False
    
    def is_variable(self, val):
        return isinstance(val, str) and val.islower() and len(val) == 1
    
    def is_constant(self, val):
        return isinstance(val, str) and len(val) >= 2 and val.islower()
    
    def mgu(self, literal1, literal2):
        begin1 = literal1.find('(')
        end1 = literal1.find(')')
        begin2 = literal2.find('(')
        end2 = literal2.find(')')
        args1 = literal1[begin1 + 1:end1].split(',')
        args2 = literal2[begin2 + 1:end2].split(',')
        if len(args1) != len(args2):
            return False
        n = len(args1)
        unification = {}
        if DEBUG:
            print(args1, args2)
        
        """
        1. 如果都是变量，则不可合一
        2. 如果都是常量，且不相等，则不可合一
        3. 当一个是变量，一个是项（在这里案例中只有常量），可以合一
        """
        while True:
            if args1 == args2:
                return unification
            for i in range(n):
                val1 = args1[i]
                val2 = args2[i]
                if self.is_variable(val1) and self.is_variable(val2):
                    return None
                elif self.is_constant(val1) and self.is_constant(val2) and val1 != val2:
                    return None
                elif self.is_variable(val1) and self.is_constant(val2):
                    unification[val1] = val2
                    args1 = [unification[val] if val in unification else val for val in args1]
                    args2 = [unification[val] if val in unification else val for val in args2]
                elif self.is_constant(val1) and self.is_variable(val2):
                    unification[val2] = val1
                    args1 = [unification[val] if val in unification else val for val in args1]
                    args2 = [unification[val] if val in unification else val for val in args2]
            
    def resolution(self):
        clauseset = self.clauses
        clauseset = list(OrderedDict.fromkeys(clauseset))
        while True:
            n = len(clauseset)
            for i in range(n):
                for j in range (i + 1, n):
                    clause1 = clauseset[i]
                    clause2 = clauseset[j]
                    for literal1 in clause1:
                        for literal2 in clause2:
                            if self.is_complement(literal1, literal2):
                                if DEBUG:
                                    print(literal1, literal2, "is complement")
                                unification = self.mgu(literal1, literal2)
                                self.substitute(unification, literal1)
                                self.substitute(unification, literal2)
            
if __name__ == "__main__":
    test1 = Sentences("test1.txt")
    test1.resolution()