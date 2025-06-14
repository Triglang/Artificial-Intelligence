DEBUG = 0

"""判断是否为变量"""
def is_variable(val):
        return isinstance(val, str) and val.islower() and len(val) == 1
    
"""判断是否为常量（小写多字母）"""
def is_constant(val):
    return isinstance(val, str) and len(val) >= 2 and val.islower()
    
"""判断是否为谓词（大写字母开头）"""
def is_predicate(val):
    return val[0].isupper()
    
"""判断是否是项"""
def is_item(val):
    return is_constant(val) or is_predicate(val)
    
"""从文字中提取参数列表"""
def get_arguments(literal):
    begin = literal.find('(')
    end = literal.rfind(')')
    return literal[begin + 1:end].split(',')

def args_substitute(args, unification):
    newargs = []
    for literal in args:
        prefix = ""
        suffix = ""
        arg = literal
        while is_predicate(arg):
            begin = arg.find('(')
            end = arg.rfind(')')
            prefix += arg[:begin + 1]
            suffix += arg[end:]
            arg = arg[begin + 1:end]
            
        if arg in unification:
            newargs.append(prefix + unification[arg] + suffix)
        else:
            newargs.append(prefix + arg + suffix)
            
    return newargs

def set_difference(arg1, arg2):
    while is_predicate(arg1) and is_predicate(arg2):
        arg1
        
    
"""计算最一般合一"""
def mgu(literal1, literal2):
    args1 = get_arguments(literal1)
    args2 = get_arguments(literal2)
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
            val1, val2 = set_difference(args1[i], args2[i])
            if is_variable(val1) and is_variable(val2):
                return None
            elif is_item(val1) and is_item(val2) and val1 != val2:
                return None
            elif is_variable(val1) and is_item(val2):
                unification[val1] = val2
                # 应用替换到整个参数列表
                args1 = [unification[val] if val in unification else val for val in args1]
                args2 = [unification[val] if val in unification else val for val in args2]
            elif is_item(val1) and is_variable(val2):
                unification[val2] = val1
                args1 = [unification[val] if val in unification else val for val in args1]
                args2 = [unification[val] if val in unification else val for val in args2]
                
if __name__ == "__main__":
    literal1 = "P(x,aa)"
    literal2 = "P(bb,y)"
    # print(mgu(literal1, literal2))
    literal3 = "P(aa,x,F(G(y)))"
    literal4 = "P(z,F(z),F(u))"
    print(mgu(literal3, literal4))