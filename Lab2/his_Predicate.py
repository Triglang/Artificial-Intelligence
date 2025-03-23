from collections import OrderedDict
DEBUG = 0
MYSELF = 1

#判断两个文字是否互补
def Complementary(literal1,literal2):
    #谓词：从文字开头到'('的子字符串
    end1 = literal1.find('(')
    end2 = literal2.find('(')
    #互补：文字间只有~的区别
    if literal2[0] == '~' and literal2[1:end2] == literal1[:end1]:
        return True
    if literal1[0] == '~' and literal1[1:end1] == literal2[:end2]:
        return True 
    return False
 
#得到题目要求的归结编号
def Index(literal_index,clause_index,length):
    if length == 1: #如果子句只有一个元素，则文字索引不再需要
        index = str(clause_index+1)
    else:           #否则将文字索引变为字母
        index = str(clause_index+1) + chr(ord('a')+literal_index)
    return index
 
#得到文字的参数列表
def Parameter(literal):
    #参数是在()内且以,分隔的多个子字符串
    start = literal.find('(')
    end = literal.find(')')
    para = literal[start+1:end].split(',')
    return para
 
#置换旧子句得到新子句
def Substitute(unification,clause):
    if unification == {}:   #字典为空，说明无需置换，直接返回原子句
        return clause
    newclause = []
    for literal in clause:
        start = literal.find('(')
        end = literal.find(')')
        para = literal[start + 1:end].split(',')   #得到文字的参数列表
        para = [unification[item] if item in unification else item for item in para]
        newliteral = literal[:start + 1] + ','.join(para) + literal[end:]  #合并成新子句
        newclause.append(newliteral)
    return tuple(newclause) #返回新子句，注意是元组格式
 
#归结：消除互补项
def Resolve(clause1,clause2,literal_index1,literal_index2):
    newclause = list(clause1) + list(clause2)
    newclause.remove(clause1[literal_index1])
    newclause.remove(clause2[literal_index2])
    newclause = list(OrderedDict.fromkeys(newclause))   #消除相同的子句
    return tuple(newclause)
 
#得到归结式
def Sequence(newclause,unification,index1,index2):
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

 
#得到归结式的子句索引
def Number(clause):
    start = clause.find('[')
    end = clause.find(']')
    number = clause[start+1:end].split(',')
    #将文字索引去掉
    num1 = int(''.join(item for item in number[0] if not item.isalpha()))
    num2 = int(''.join(item for item in number[1] if not item.isalpha()))
    return num1,num2
 
#得到新归结式的子句索引
def Renumber(num,result,useful_process,size):
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
def Resequence(sequence,num1,num2,newnum1,newnum2):
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
def Simplify(result,size):
    base_process = result[0:size+1] #初始子句集
    useful_process = []             #有用子句集
    number = [len(result)-1]        #用作队列，先将空子句的索引入列
    while number != []:
        # print(number)
        number0 = number.pop(0)                 #提取队列首元素，即有用子句的索引
        useful_process.append(result[number0])  #将有用子句加入到有用子句集            
        num1,num2 = Number(result[number0])     #得有用子句用到的亲本子句索引
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
        num1,num2 = Number(useful_process[i])
        newnum1 = str(Renumber(num1,result,useful_process,size))
        newnum2 = str(Renumber(num2,result,useful_process,size))
        useful_process[i] = Resequence(useful_process[i],str(num1),str(num2),newnum1,newnum2)
    return useful_process
 
#打印结果
def Print(result):
    print(result[0])
    for i in range(1,len(result)):
        print(i,result[i])

import re

class Sentences:
    # def __init__(self, path):
    #     self.clauses = []
    #     self.result = []
    #     with open(path, 'r') as f:
    #         lines = [line.strip() for line in f]
            
    #     for line in lines:
    #         if not line or line.strip().lower() == 'kb:' or line.strip().lower() == 'query:': continue
    #         clause = self.parse_clause(line)
    #         if clause: self.clauses.append(clause)
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
            self.clauses.append(literals)

    def parse_clause(self, line):
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            content = line[1:-1].strip()[:-1]
        else:
            content = line[:-1]
        
        literals = []
        for lit in content.split('),'):
            lit = lit.strip() + ')'
            # 替换参数中的空格（如"L(tony, rain)" → "L(tony,rain)"）
            lit = re.sub(r',\s+', ',', lit)
            literals.append(lit)
        
        return tuple(literals) if literals else None
    
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

    #得到合一
    def Unify(self, literal1,literal2):
        if MYSELF:
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
        else:
            variable = {'u','v','w','x','y','z','uu','vv','ww','xx','yy','zz'}  #变量集
            unification = {}
            if para1 == para2:  #若一开始就相等，直接返回空合一
                return unification 
            while True:
                for i,j in zip(para1,para2):
                    if i in variable and j in variable: #若都是变量，不可合一
                        return None
                    elif i not in variable and j not in variable and i != j:    #若都是常量且不相等，不可合一
                        return None
                    elif i in variable and j not in variable:
                        unification[i] = j
                        break
                    elif j in variable and i not in variable:
                        unification[j] = i
                        break
                #合一复合
                para1 = [unification[item] if item in unification else item for item in para1]
                para2 = [unification[item] if item in unification else item for item in para2]
                if para1 == para2:
                    break
            return unification
    
    #反演：支持集策略
    def Refutation(self):
        clauseset = list(self.clauses)            #拷贝一份，防止更改原初始子句集
        supportset = [self.clauses[-1]]           #支持集，默认self.clauses最后一个元素是目标子句的否定
        result = ['归结顺序:'] + self.clauses      #将0位置补充元素，确保编号和列表索引对应
        while True:
            new_clauseset = []  #此级得到的全部的新的子句
            clause_index1 = 0
            for clause1 in clauseset:
                if clause1 in self.clauses:
                    clause_index2 = 0
                    for clause2 in clauseset:
                        if clause1 != clause2 and clause2 in supportset: #其中一个亲本子句必须来自支持集
                            literal_index1 = 0
                            for literal1 in clause1:
                                literal_index2 = 0
                                for literal2 in clause2:
                                    #判断是否互补
                                    if Complementary(literal1,literal2):
                                        #得到索引
                                        index1 = Index(literal_index1,clause_index1,len(clause1))
                                        index2 = Index(literal_index2,clause_index2,len(clause2))
                                        #得到互补文字
                                        literal1 = clause1[literal_index1]
                                        literal2 = clause2[literal_index2]
                                        #得到参数列表
                                        para1 = Parameter(literal1)
                                        para2 = Parameter(literal2)
                                        #得到合一置换
                                        if MYSELF:
                                            unification = self.Unify(literal1,literal2)
                                        else:
                                            unification = self.Unify(para1,para2)
                                        #合一置换不存在则退出
                                        if unification == None:
                                            break
                                        #得到子句
                                        newclause1 = Substitute(unification,clause1)    
                                        newclause2 = Substitute(unification,clause2)  
                                        #得到归结子句
                                        newclause = Resolve(newclause1,newclause2,literal_index1,literal_index2)
                                        #归结子句存在于原子句集则退出
                                        if any([set(newclause)==(set(item)) for item in clauseset]):
                                            break
                                        #归结子句存在于新子句集则退出
                                        if any([set(newclause)==(set(item)) for item in new_clauseset]):
                                            break
                                        #得到新归结式
                                        sequence = Sequence(newclause,unification,index1,index2)
                                        #加入结果列表和新子句列表
                                        result.append(sequence)
                                        new_clauseset.append(newclause)
                                        #当且仅当新子句为空，则可退出函数，返回结果列表
                                        if newclause == ():
                                            return result
                                    literal_index2 += 1
                                literal_index1 += 1
                        clause_index2 += 1
                clause_index1 += 1
            clauseset += new_clauseset      #更新子句列表
            supportset += new_clauseset     #更新支持集
    
    def resolution(self):
        self.result = self.Refutation()
        
    def reindex(self):
        new_result = Simplify(self.result,len(self.clauses))
        Print(new_result)

if __name__ == "__main__":
    test1 = Sentences("test1.txt")
    test1.resolution()
    test1.reindex()
