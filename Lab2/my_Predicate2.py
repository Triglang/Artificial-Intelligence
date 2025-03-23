class Sentences:
    def __init__(self, path):
        self.clauses = []  # 存储所有子句的列表
        with open(path, 'r') as f:
            lines = [line.strip() for line in f]

        # 分割KB和QUERY部分
        kb_lines = []
        query_lines = []
        current_section = None
        for line in lines:
            lower_line = line.lower()
            if lower_line.startswith('kb:'):
                current_section = 'kb'
            elif lower_line.startswith('query:'):
                current_section = 'query'
            else:
                if current_section == 'kb':
                    kb_lines.append(line)
                elif current_section == 'query':
                    query_lines.append(line)

        # 解析KB子句
        for line in kb_lines:
            if line:
                clause = self.parse_clause(line)
                if clause is not None:
                    self.clauses.append(clause)

        # 解析查询
        for line in query_lines:
            if line:
                clause = self.parse_clause(line)
                if clause is not None:
                    self.clauses.append(clause)
        
        # 检查空子句
        if False in self.clauses:
            print("空子句已存在于初始知识库中。")

    def parse_clause(self, line):
        line = line.strip()
        if not line:
            return None
        if line == '∅' or line.lower() == 'false':
            return False
        if line.startswith('(') and line.endswith(')'):
            content = line[1:-1].strip()
        else:
            content = line
        literals = [lit.strip() for lit in content.split(',')]
        literals = [lit for lit in literals if lit]
        if not literals:
            return False
        return tuple(literals)

    def resolution(self):
        # 预处理子句，转换为集合并过滤永真式
        clauses = set()
        for clause in self.clauses:
            if clause is False:
                clauses.add(False)
                continue
            unique = set(clause)
            sorted_clause = tuple(sorted(unique))
            if self.is_tautology(sorted_clause):
                continue
            clauses.add(sorted_clause)
        
        # 初始检查空子句
        if False in clauses:
            return True, [clauses]
        
        steps = [set(clauses)]  # 记录每一步的子句集合
        
        while True:
            new_clauses = set()
            # clause_list = list(clauses)
            clause_list = self.clauses
            # 遍历所有可能的子句对
            for i in range(len(clause_list)):
                for j in range(i + 1, len(clause_list)):
                    c1 = clause_list[i]
                    c2 = clause_list[j]
                    resolvents = self.resolve(c1, c2)
                    for res in resolvents:
                        if res is False:
                            new_clauses.add(False)
                            clauses.update(new_clauses)
                            steps.append(set(clauses))
                            return True, steps
                        if res not in clauses and res not in new_clauses:
                            new_clauses.add(res)
            if not new_clauses:
                return False, steps
            clauses.update(new_clauses)
            steps.append(set(clauses))

    def is_complement(self, lit1, lit2):
        pred1, args1, neg1 = self.parse_literal(lit1)
        pred2, args2, neg2 = self.parse_literal(lit2)
        return pred1 == pred2 and neg1 != neg2

    def is_tautology(self, clause):
        if clause is False:
            return False
        for lit in clause:
            complement = self.get_complement(lit)
            if complement in clause:
                return True
        return False

    def parse_literal(self, literal):
        negated = False
        if literal.startswith('~'):
            negated = True
            literal = literal[1:]
        parts = literal.split('(')
        predicate = parts[0]
        args = parts[1].strip(')').split(',') if len(parts) > 1 else []
        return predicate, args, negated

    def mgu(self, term1, term2, substitution=None):
        if substitution is None:
            substitution = {}
        # stack = [(term1, term2)]
        deque1 = term1
        deque2 = term2
        while deque1 and deque2:
            t1 = deque1.pop(0)
            t2 = deque2.pop(0)
            # 步骤1：如果当前项在替换下已相同，继续处理下一个差异
            t1_sub = substitution.get(t1, t1) if self.is_variable(t1) else t1
            t2_sub = substitution.get(t2, t2) if self.is_variable(t2) else t2

            if t1_sub == t2_sub:
                continue
            # 步骤2：处理变量和项的情况
            if self.is_variable(t1_sub):
                # 检查 t2_sub 是否包含 t1_sub (防止循环)
                if self.occurs_check(t1_sub, t2_sub, substitution):
                    return None
                # 添加替换 t1_sub → t2_sub，并更新所有现有替换
                substitution = self.compose_substitution(substitution, {t1_sub: t2_sub})
            elif self.is_variable(t2_sub):
                if self.occurs_check(t2_sub, t1_sub, substitution):
                    return None
                substitution = self.compose_substitution(substitution, {t2_sub: t1_sub})
            # 步骤3：处理谓词/函数项
            elif isinstance(t1_sub, list) and isinstance(t2_sub, list):
                if t1_sub[0] != t2_sub[0] or len(t1_sub) != len(t2_sub):
                    return None  # 函数名或参数数量不同
                # 将参数对推入栈（从右到左，确保顺序）
                for p1, p2 in zip(reversed(t1_sub[1:]), reversed(t2_sub[1:])):
                    stack.append((p1, p2))
            else:
                return None  # 类型不匹配
        return substitution

    def is_variable(self, term):
        """判断是否为变量（单个小写字母）"""
        b1 = isinstance(term, str)
        b2 = bool()
        b3 = bool()
        if b1:
            b2 = (len(term) == 1)
        if b2:
            b3 = term.islower()
        return b1 and b2 and b3

    def occurs_check(self, var, term, substitution):
        """检查变量 var 是否出现在 term 中（考虑替换链）"""
        term = substitution.get(term, term)
        if var == term:
            return True
        if isinstance(term, list):
            return any(self.occurs_check(var, t, substitution) for t in term)
        return False

    def compose_substitution(self, old_sub, new_sub):
        """合并替换：先应用 old_sub，再应用 new_sub"""
        combined = {}
        # 应用 new_sub 到 old_sub 的值
        for key in old_sub:
            combined[key] = self.apply_substitution_single(old_sub[key], new_sub)
        # 添加 new_sub 的替换
        for key in new_sub:
            if key not in combined:
                combined[key] = new_sub[key]
        # 应用 new_sub 到 combined 的值
        for key in combined:
            combined[key] = self.apply_substitution_single(combined[key], new_sub)
        return combined

    def apply_substitution_single(self, term, substitution):
        """对单个项应用替换"""
        if self.is_variable(term) and term in substitution:
            return substitution[term]
        elif isinstance(term, list):
            return [term[0]] + [self.apply_substitution_single(t, substitution) for t in term[1:]]
        return term
    
    def is_constant(self, term):
        return isinstance(term, str) and len(term) >= 2 and term.islower()

    def unify_var(self, var, term, substitution):
        if var in substitution:
            return self.mgu(substitution[var], term, substitution)
        elif self.is_variable(term) and term in substitution:
            return self.mgu(var, substitution[term], substitution)
        elif self.is_constant(term) or self.is_variable(term):
            substitution[var] = term
            return substitution
        else:
            return None

    def apply_substitution(self, clause, substitution):
        new_clause = []
        for lit in clause:
            pred, args, neg = self.parse_literal(lit)
            new_args = [substitution.get(arg, arg) for arg in args]
            new_lit = ('~' if neg else '') + pred + '(' + ','.join(new_args) + ')'
            new_clause.append(new_lit)
        return tuple(new_clause)

    def resolve(self, c1, c2):
        resolvents = []
        if c1 is False or c2 is False:
            return []
        # 变量标准化：确保两个子句变量独立
        c1 = self.standardize_variables(c1, 'x')
        c2 = self.standardize_variables(c2, 'a')
        for lit1 in c1:
            for lit2 in c2:
                if self.is_complement(lit1, lit2):
                    pred1, args1, _ = self.parse_literal(lit1)
                    pred2, args2, _ = self.parse_literal(lit2)
                    substitution = self.mgu(args1, args2)
                    if substitution is not None:
                        new_c1 = self.apply_substitution(c1, substitution)
                        new_c2 = self.apply_substitution(c2, substitution)
                        for new_lit1 in new_c1:
                            for new_lit2 in new_c2:
                                if self.is_complement(new_lit1, new_lit2):
                                    # new_lit1_tmp 和 new_lit2_tmp 中消去 new_lit1 和 new_lit2，然后结束两层循环
                                    new_lits = [lit for lit in new_c1 if lit != new_lit1] + [lit for lit in new_c2 if lit != new_lit2]
                                          
                                    unique_lits = list(set(new_lits))
                                    sorted_lits = tuple(sorted(unique_lits))
                                    if not self.is_tautology(sorted_lits):
                                        if len(sorted_lits) == 0:
                                            resolvents.append(False)
                                        else:
                                            resolvents.append(sorted_lits)
                                    
                                    return resolvents
        return resolvents

    def standardize_variables(self, clause, start_char):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        start_index = letters.index(start_char.lower())
        var_mapping = {}
        new_clause = []
        for lit in clause:
            pred, args, neg = self.parse_literal(lit)
            new_args = []
            for arg in args:
                if self.is_variable(arg):
                    if arg not in var_mapping:
                        var_mapping[arg] = letters[start_index % 26]
                        start_index += 1
                    new_args.append(var_mapping[arg])
                else:
                    new_args.append(arg)
            new_lit = ('~' if neg else '') + pred + '(' + ','.join(new_args) + ')'
            new_clause.append(new_lit)
        return tuple(new_clause)

    def get_complement(self, lit):
        if lit.startswith('~'):
            return lit[1:]
        else:
            return '~' + lit

if __name__ == '__main__':
    test1 = Sentences('test1.txt')
    result, steps = test1.resolution()
    print("归结结果:", "成功" if result else "失败")
    for i, step in enumerate(steps):
        print(f"步骤 {i+1}: {step}")