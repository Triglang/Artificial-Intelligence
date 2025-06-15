from collections import OrderedDict

DEBUG = 0       # 调试模式开关

class Sentences:
    def __init__(self, path):
        self.clauses = []       # 存储所有子句
        self.step = []          # 存储归结步骤记录
        with open(path, 'r') as f:
            lines = [line.strip() for line in f]
            
        for line in lines:
            # 跳过空行和标题行
            if not line or line.lower() == "kb:" or line.lower() == "query:":
                continue
            
            # 分隔文字
            if line.startswith('(') and line.endswith(')'):
                line = line[1: -1]
            line = line[:-1]
            literals = []
            
            for lit in line.split("),"):
                literals.append(lit.replace(" ", "") + ")")
            self.clauses.append(tuple(literals))
        
        if DEBUG:   # 调试输出
            for item in self.clauses:
                print(item)
            print(self.clauses)
            
    """检查两个文字是否为互补对"""
    def is_complement(self, literal1, literal2):
        # if DEBUG:
        #     print("literal1 and literal2: ", type(literal1), type(literal2))
        end1 = literal1.find('(')
        end2 = literal2.find('(')
        if literal1.startswith('~') and literal1[1:end1] == literal2[:end2]:
            return True
        elif literal2.startswith('~') and literal1[:end1] == literal2[1:end2]:
            return True
        
        return False
    
    """判断是否为变量"""
    def is_variable(self, val):
        return isinstance(val, str) and val.islower() and len(val) == 1
    
    """判断是否为常量（小写多字母）"""
    def is_constant(self, val):
        return isinstance(val, str) and len(val) >= 2 and val.islower()
    
    """判断是否为谓词（大写字母开头）"""
    def is_predicate(self, val):
        return val[0].isupper()
    
    """判断是否是项"""
    def is_item(self, val):
        return self.is_constant(val) or self.is_predicate(val)
    
    """从文字中提取参数列表"""
    def get_arguments(self, literal):
        begin = literal.find('(')
        end = literal.rfind(')')
        return literal[begin + 1:end].split(',')
    
    """计算最一般合一"""
    def mgu(self, literal1, literal2):
        args1 = self.get_arguments(literal1)
        args2 = self.get_arguments(literal2)
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
                elif self.is_item(val1) and self.is_item(val2) and val1 != val2:
                    return None
                elif self.is_variable(val1) and self.is_item(val2):
                    unification[val1] = val2
                    # 应用替换到整个参数列表
                    args1 = [unification[val] if val in unification else val for val in args1]
                    args2 = [unification[val] if val in unification else val for val in args2]
                elif self.is_item(val1) and self.is_variable(val2):
                    unification[val2] = val1
                    args1 = [unification[val] if val in unification else val for val in args1]
                    args2 = [unification[val] if val in unification else val for val in args2]
                    

    """应用合一替换到子句"""
    def substitute(self, unification, clause):
        newclause = []
        for literal in clause:
            args = self.get_arguments(literal)
            args = [unification[val] if val in unification else val for val in args]
            begin = literal.find('(')
            end = literal.find(')')
            newliteral = literal[:begin + 1] + ','.join(args) + literal[end:]
            newclause.append(newliteral)
        return tuple(newclause)
    
    """执行归结操作"""
    def resolve(self, clause1, clause2, literal1_index, literal2_index):
        newclause = list(clause1 + clause2)
        newclause.remove(clause1[literal1_index])
        newclause.remove(clause2[literal2_index])
        newclause = list(OrderedDict.fromkeys(newclause))
            
        return tuple(newclause)
    
    def index(self, literal_index,clause_index,length):
        if length == 1: #如果子句只有一个元素，则文字索引不再需要
            index = str(clause_index+1)
        else:           #否则将文字索引变为字母
            index = str(clause_index+1) + chr(ord('a')+literal_index)
        return index
    
    """生成文字索引标识"""
    def sequence(self, newclause,unification,index1,index2):
        string = ''
        if unification == {}:    #如果字典为空，说明不需要输出合一
            string += 'R[' + index1 + ',' + index2 + '] = '
        else:
            string += 'R[' + index1 + ',' + index2 + ']{'
            for key,value in unification.items():
                string += key + '=' + value + ','
            string = string[:-1]
            string += '} = '
        string += str(newclause)
        return string
    
    # 支持集策略下的归结推理
    def resolution(self):
        clauseset = self.clauses
        clauseset = list(OrderedDict.fromkeys(clauseset))       # 去重
        step = ['归结顺序:'] + self.clauses      #将0位置补充元素，确保编号和列表索引对应
        supportset = [clauseset[-1]]
        while True:
            clauseset_len = len(clauseset)
            new_clauseset = []
            
            # 遍历子句集
            for clause1_index in range(clauseset_len):
                for clause2_index in range (clause1_index + 1, clauseset_len):
                    clause1 = clauseset[clause1_index]
                    clause2 = clauseset[clause2_index]
                    clause1_len = len(clause1)
                    clause2_len = len(clause2)
                    if clause1 not in supportset and clause2 not in supportset:
                        continue
                    for literal1_index in range(clause1_len):
                        for literal2_index in range(clause2_len):
                            literal1 = clause1[literal1_index]
                            literal2 = clause2[literal2_index]

                            # 判断是否为互补对
                            if self.is_complement(literal1, literal2):
                                if DEBUG:
                                    print(literal1, literal2, "is complement")
                                    
                                # 最一般合一项
                                unification = self.mgu(literal1, literal2)
                                if unification == None:
                                    break
                                if DEBUG:   # 未实现功能：若违法，则跳出循环
                                    if unification == False:
                                        print("谓词的参数个数必须相同")
                                        return False
                                
                                # 最一般合一替换
                                newclause1 = self.substitute(unification, clause1)
                                newclause2 = self.substitute(unification, clause2)
                                
                                # 归结
                                newclause =  self.resolve(newclause1, newclause2, literal1_index, literal2_index)
                                
                                # 检查是否为新子句
                                if newclause in clauseset or newclause in new_clauseset:
                                    break
                                new_clauseset.append(newclause)
                                
                                # 记录步骤
                                index1 = self.index(literal1_index, clause1_index, clause1_len)
                                index2 = self.index(literal2_index, clause2_index, clause2_len)
                                sequence = self.sequence(newclause, unification, index1, index2)
                                
                                step.append(sequence)
                                
                                # 发现空子句则成功
                                if newclause == ():
                                    self.step = step
                                    return
                            literal2_index += 1
                        literal1_index += 1
            if new_clauseset:
                clauseset += new_clauseset
                supportset += new_clauseset
            else:
                return False

    #得到归结式的子句索引
    def Number(self, clause):
        start = clause.find('[')
        end = clause.find(']')
        number = clause[start+1:end].split(',')
        #将文字索引去掉
        num1 = int(''.join(item for item in number[0] if not item.isalpha()))
        num2 = int(''.join(item for item in number[1] if not item.isalpha()))
        return num1,num2
    
    #得到新归结式的子句索引
    def Renumber(self, num,result,useful_process,size):
        if num <= size: #如果是初始子句集的，直接返回
            return num
        #找到亲本子句
        sequence = result[num]
        begin = sequence.find('(')
        aim_clause = sequence[begin:]
        #找到亲本子句在化简子句集的编号
        for i in range(size+1,len(useful_process)):
            begin = useful_process[i].find('(')
            if useful_process[i][begin:] == aim_clause:
                return i
            
    #更新归结式
    def Resequence(self, sequence,num1,num2,newnum1,newnum2):
        # 第一次替换：替换第一个编号
        start = sequence.find(num1)
        end = start + len(num1)
        sequence = sequence[:start] + newnum1 + sequence[end:]
        # 第二次替换：替换第二个编号
        end = start + len(newnum1)
        start = sequence.find(num2, end)
        end = start + len(num2)
        sequence = sequence[:start] + newnum2 + sequence[end:]
        return sequence

    #化简归结过程
    def Simplify(self, result,size):
        base_process = result[0:size+1] #初始子句集
        useful_process = []             #有用子句集
        number = [len(result)-1]        #用作队列，先将空子句的索引入列
        while number != []:
            # print(number)
            number0 = number.pop(0)                 #提取队列首元素，即有用子句的索引
            useful_process.append(result[number0])  #将有用子句加入到有用子句集            
            num1,num2 = self.Number(result[number0])     #得有用子句用到的亲本子句索引
            #如果是初始子句集就无需加入
            if num1 > size:
                number.append(num1)
            if num2 > size:
                number.append(num2)
        #得到新的归结过程
        useful_process.reverse()
        useful_process = base_process + useful_process
        #将归结过程重新编号
        for i in range(size+1,len(useful_process)): 
            num1,num2 = self.Number(useful_process[i])
            newnum1 = str(self.Renumber(num1,result,useful_process,size))
            newnum2 = str(self.Renumber(num2,result,useful_process,size))
            useful_process[i] = self.Resequence(useful_process[i],str(num1),str(num2),newnum1,newnum2)
        return useful_process

    def reindex(self):
        if DEBUG:
            for item in self.step:
                print(item)
        new_result = self.Simplify(self.step,len(self.clauses))        
        print(new_result[0])
        for i in range(1,len(new_result)):
            print(i,new_result[i])
            
if __name__ == "__main__":
    test1 = Sentences("test1.txt")
    test1.resolution()
    test1.reindex()