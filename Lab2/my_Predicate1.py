class Sentences:  # 确保类名与调用一致
    def __init__(self, path):
        self.clauses = []
        self.query = None
        self.seen_clauses = set()  # 用于去重的哈希存储
        self.steps = []  # 新增步骤记录

        with open(path, 'r', encoding='utf-8') as f:
            content = [line.strip() for line in f if line.strip()]

        section = None
        for line in content:
            if line == "KB:":
                section = "KB"
                continue
            elif line == "QUERY:":
                section = "QUERY"
                continue

            if section == "KB":
                # 解析子句并去重
                clause = self._parse_clause(line)[0]
                clause_hash = self._hash_clause(clause)
                
                if clause_hash not in self.seen_clauses:
                    self.seen_clauses.add(clause_hash)
                    self.clauses.append(clause)
                    
            elif section == "QUERY":
                self.query = self._parse_clause(line)[0]

        # 添加查询子句
        if self.query:
            self.clauses.append(self.query)
            
        for clause in self.clauses:
            print(clause)
            
    def _hash_clause(self, clause):
        """生成子句的唯一哈希值"""
        normalized = []
        for lit in sorted(clause, key=lambda x: (x['predicate'], tuple(x['args']))):
            # 参数排序处理（如需变量标准化可在此扩展）
            args = sorted(lit['args']) if all(a.islower() for a in lit['args']) else lit['args']
            normalized.append((
                lit['predicate'],
                tuple(args),
                lit['negated']
            ))
        return hash(tuple(normalized))

    def _parse_clause(self, clause_str):
        """修正后的子句解析方法"""
        clause_str = clause_str.strip()
        
        # 移除外层括号（如果存在）
        if clause_str.startswith('(') and clause_str.endswith(')'):
            clause_str = clause_str[1:-1]

        clauses = []
        current = []
        stack = []
        
        # 增强版分割逻辑
        for char in clause_str:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if stack:
                    stack.pop()
            
            # 仅当栈为空时分割顶层逗号
            if char == ',' and not stack:
                lit = ''.join(current).strip()
                if lit:
                    clauses.append(self._parse_literal(lit))
                current = []
            else:
                current.append(char)
        
        # 处理最后一个文字
        if current:
            lit = ''.join(current).strip()
            clauses.append(self._parse_literal(lit))
        
        return [clauses]  # 返回包含多个文字的列表

    def _parse_literal(self, literal_str):  # 添加缺失的方法
        """ 解析单个谓词文字 """
        negated = False
        if literal_str.startswith('~'):
            negated = True
            literal_str = literal_str[1:]
        
        # 分离谓词和参数
        pred_start = literal_str.find('(')
        if pred_start == -1:
            return {
                'predicate': literal_str,
                'args': [],
                'negated': negated
            }
        
        predicate = literal_str[:pred_start]
        args_str = literal_str[pred_start+1:-1]
        args = []
        
        # 解析带嵌套的参数
        current_arg = []
        stack = []
        for char in args_str:
            if char == '(':
                stack.append(char)
            elif char == ')':
                stack.pop()
            
            if char == ',' and not stack:
                args.append(''.join(current_arg).strip())
                current_arg = []
            else:
                current_arg.append(char)
        if current_arg:
            args.append(''.join(current_arg).strip())
        
        return {
            'predicate': predicate,
            'args': args,
            'negated': negated
        }
        
    def resolution(self):
        """执行归结过程并记录步骤"""
        from collections import deque

        # 初始化队列和工作集合
        queue = deque(self.clauses)
        known_clauses = {self._hash_clause(c): c for c in self.clauses}
        self.steps = []

        step_counter = 0
        while queue:
            # 取出两个不同子句
            clause1 = queue.popleft()
            for clause2 in list(known_clauses.values()):
                if clause1 == clause2:
                    continue

                # 标准化变量
                std_clause1 = self.standardize_vars(clause1, prefix='x')
                std_clause2 = self.standardize_vars(clause2, prefix='y')

                # 查找可归结的文字对
                for lit1 in std_clause1:
                    for lit2 in std_clause2:
                        if lit1['negated'] != lit2['negated']:
                            continue
                        if lit1['predicate'] != lit2['predicate']:
                            continue

                        # 尝试合一
                        substitution = self.unify(
                            lit1['args'], lit2['args']
                        )
                        if substitution is None:
                            continue

                        # 生成新子句
                        new_clause = self._resolve(
                            std_clause1, std_clause2, 
                            lit1, lit2, substitution
                        )

                        # 记录步骤
                        step_counter += 1
                        self.steps.append({
                            'step': step_counter,
                            'parents': (clause1, clause2),
                            'unified_lits': (lit1, lit2),
                            'substitution': substitution,
                            'new_clause': new_clause
                        })

                        # 发现空子句
                        if not new_clause:
                            print("归结成功！发现矛盾")
                            return True

                        # 添加新子句
                        clause_hash = self._hash_clause(new_clause)
                        if clause_hash not in known_clauses:
                            known_clauses[clause_hash] = new_clause
                            queue.append(new_clause)

        print("无法归结出空子句")
        return False

    def _resolve(self, c1, c2, lit1, lit2, sub):
        """生成归结后的新子句"""
        # 应用变量替换
        resolved = []
        for lit in c1:
            if lit != lit1:
                resolved.append(self.apply_substitution(lit.copy(), sub))
        for lit in c2:
            if lit != lit2:
                resolved.append(self.apply_substitution(lit.copy(), sub))

        # 去除重复文字
        seen = set()
        unique = []
        for lit in resolved:
            lit_hash = (lit['predicate'], tuple(lit['args']), lit['negated'])
            if lit_hash not in seen:
                seen.add(lit_hash)
                unique.append(lit)
        return unique

    def unify(self, args1, args2):
        """合一算法实现"""
        if len(args1) != len(args2):
            return None

        substitution = {}
        for a1, a2 in zip(args1, args2):
            if a1 == a2:
                continue

            # 处理变量替换
            if self.is_variable(a1):
                substitution[a1] = a2
            elif self.is_variable(a2):
                substitution[a2] = a1
            else:
                return None  # 常量不匹配

        return substitution

    @staticmethod
    def is_variable(term):
        """判断是否为变量（假设小写字母为变量）"""
        return term.islower()

    def standardize_vars(self, clause, prefix='x'):
        """变量标准化"""
        var_map = {}
        new_clause = []
        for lit in clause:
            new_lit = lit.copy()
            new_args = []
            for arg in lit['args']:
                if self.is_variable(arg):
                    if arg not in var_map:
                        var_map[arg] = f"{prefix}_{len(var_map)}"
                    new_args.append(var_map[arg])
                else:
                    new_args.append(arg)
            new_lit['args'] = new_args
            new_clause.append(new_lit)
        return new_clause

    def apply_substitution(self, lit, substitution):
        """应用替换到单个文字"""
        new_args = []
        for arg in lit['args']:
            new_args.append(substitution.get(arg, arg))
        lit['args'] = new_args
        return lit

    def print_steps(self):
        """打印归结过程"""
        for step in self.steps:
            print(f"步骤 {step['step']}:")
            print(f"  父类1: {self._clause_to_str(step['parents'][0])}")
            print(f"  父类2: {self._clause_to_str(step['parents'][1])}")
            print(f"  消解文字: {self._lit_to_str(step['unified_lits'][0])} 和 {self._lit_to_str(step['unified_lits'][1])}")
            print(f"  替换: {step['substitution']}")
            print(f"  生成子句: {self._clause_to_str(step['new_clause'])}")
            print()

    def _clause_to_str(self, clause):
        """子句可视化"""
        if not clause:
            return "□"
        return " ∨ ".join([self._lit_to_str(lit) for lit in clause])

    def _lit_to_str(self, lit):
        """文字可视化"""
        pred = lit['predicate']
        args = ",".join(lit['args'])
        neg = "¬" if lit['negated'] else ""
        return f"{neg}{pred}({args})"