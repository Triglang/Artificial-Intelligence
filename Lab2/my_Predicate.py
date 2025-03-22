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

        # 解析查询并处理其否定形式
        for line in query_lines:
            if line:
                clause = self.parse_clause(line)
                if clause is not None:
                    self.clauses.append(clause)
        
        # for item in self.clauses:
        #     print(item)
        # print(self.clauses)

    def parse_clause(self, line):
        line = line.strip()
        if not line:
            return None
        if line == '∅' or line.lower() == 'false':
            return False
        # 处理括号
        if line.startswith('(') and line.endswith(')'):
            content = line[1:-1].strip()
        else:
            content = line
        if not content:
            return False
        # 分割并清理每个文字
        literals = [lit.strip() for lit in content.split(',')]
        literals = [lit for lit in literals if lit]
        if not literals:
            return False
        return tuple(literals)