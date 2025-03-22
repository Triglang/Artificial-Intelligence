class Sentences:  # 确保类名与调用一致
    def __init__(self, path):
        self.clauses = []
        self.query = None
        self.seen_clauses = set()  # 用于去重的哈希存储

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