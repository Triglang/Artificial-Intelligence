from collections import OrderedDict
 
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
 
#得到合一
def Unify(para1,para2):
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
 
#反演：支持集策略
def Refutation(KB):
    clauseset = list(KB)            #拷贝一份，防止更改原初始子句集
    supportset = [KB[-1]]           #支持集，默认KB最后一个元素是目标子句的否定
    result = ['归结顺序:'] + KB      #将0位置补充元素，确保编号和列表索引对应
    while True:
        new_clauseset = []  #此级得到的全部的新的子句
        clause_index1 = 0
        for clause1 in clauseset:
            if clause1 in KB:
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
                                    unification = Unify(para1,para2)
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
 
#归结反演
def ResolutionFOL(KB):
    result = Refutation(KB)
    new_result = Simplify(result,len(KB))
    Print(new_result)
    
KB1 = [('GradStudent(sue)',),('~GradStudent(x)','Student(x)'),('~Student(x)','HardWorker(x)'),
     ('~HardWorker(sue)',)]
 
KB2 = [('On(tony,mike)',),('On(mike,john)',),('Green(tony)',),('~Green(john)',),
      ('~On(xx,yy)','~Green(xx)','Green(yy)')]
 
KB3 = [('A(tony)',), ('A(mike)',), ('A(john)',), ('L(tony,rain)',), ('L(tony,snow)',),
      ('~A(x)', 'S(x)', 'C(x)'), ('~C(y)', '~L(y,rain)'), ('L(z,snow)', '~S(z)'),
      ('~L(tony,u)', '~L(mike,u)'), ('L(tony,v)', 'L(mike,v)'), ('~A(w)', '~C(w)', 'S(w)')]
 
ResolutionFOL(KB1)
ResolutionFOL(KB2)
ResolutionFOL(KB3)