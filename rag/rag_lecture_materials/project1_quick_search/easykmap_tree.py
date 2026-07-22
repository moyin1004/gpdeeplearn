#%%
# 配置项
CACHED = False
CACHED_SIZE = 10  # 缓存大小，需 >= search中的limit
CACHED_THREHOLD = 10  # 被搜索超过多少次后才加入缓存

# Node 类：是树的基本单元，每个节点存储一个字符（key）--这个概念很重要，本质上就是字符级别的不断深入与匹配(树查找的本质)，是否为叶子节点（is_leaf），权重（weight），
# 以及一个缓存列表（cache）用于存储排序后的结果。此外，还记录了该节点的搜索次数（search_count）。
class Node(dict):
    def __init__(self, key, is_leaf=False, weight=0, **kwargs):
        """
        @param key: 节点字符
        @param is_leaf: 是否叶子节点
        @param weight: 节点权重（热度）
        @param kwargs: 其他任意参数
        """
        self.key = key
        self.is_leaf = is_leaf
        self.weight = weight
        self.cache = []  # 缓存排序后的结果
        self.search_count = 0  # 搜索计数
        # 设置额外属性
        for k, v in kwargs.items():
            setattr(self, k, v)
    def __str__(self):
        return f'<Node key:{self.key} is_leaf:{self.is_leaf} weight:{self.weight} Subnodes: {self.keys()}>'
    def add_subnode(self, node):
        """添加子节点"""
        self[node.key] = node
    def get_subnode(self, key):
        """获取子节点"""
        return self.get(key)
    def has_subnode(self):
        """判断是否有子节点"""
        return len(self) > 0
    def get_top_node(self, prefix):
        """获取前缀的最后一个节点"""
        top = self
        for k in prefix:
            # 获取子节点
            top = top.get_subnode(k)
            if top is None:
                return None
        return top

 # 深度优先遍历:depth_walk 函数：从给定节点开始，递归遍历所有子节点，返回所有叶子节点及其路径。路径由从根节点到叶子节点的字符拼接而成。
def depth_walk(node):
    """
    深度优先遍历，返回所有叶子节点及其路径
    """
    result = []
    if node.is_leaf:
        result.append(('', node))
    # print("node.item():",node.items())
    """
    Search 'ba':
    node.item(): dict_items([('n', {'a': {'n': {'a': {}}}, 'd': {}})]) 
    node.item(): dict_items([('a', {'n': {'a': {}}}), ('d', {})]) 
    node.item(): dict_items([('n', {'a': {}})])
    node.item(): dict_items([('a', {})])
    node.item(): dict_items([])
    """
    for key, subnode in node.items():
        #如果匹配上了当前字符key,则再调用深度遍历,寻找下一个字符的匹配项(子节点),层层深入
        for subkey, snode in depth_walk(subnode):
            result.append((key + subkey, snode))
            # print("subkey:",subkey,"snode:",snode) #subkey: d snode: <Node key:d is_leaf:True weight:70 Subnodes: dict_keys([])>

    # print("result:",result) #result: [('nana', {}), ('nd', {})]

    return result

# search 函数：根据前缀搜索匹配的关键词，并按权重降序排列结果。它首先通过 get_top_node 方法找到前缀的最后一个节点，然后根据缓存机制决定是否直接返回缓存结果。
# 如果没有缓存，则通过 depth_walk 获取所有匹配结果并排序。如果搜索次数达到阈值（CACHED_THREHOLD），则将结果存入缓存。
def search(node, prefix, limit=None, is_case_sensitive=False):
    """
    搜索前缀匹配的所有结果，按权重降序排列
    """
    if not is_case_sensitive:
        prefix = prefix.lower()
    #获取前缀的最后一个节点
    top_node = node.get_top_node(prefix)
    if top_node is None:
        return []
    # 搜索计数
    top_node.search_count += 1
    # 使用缓存（如果启用且存在）,启用缓存查询的通常是高频的数据
    if CACHED and top_node.cache:
        return top_node.cache[:limit] if limit else top_node.cache
    # 获取所有匹配结果
    results = [(prefix + subkey, pnode) for subkey, pnode in depth_walk(top_node)]
    # 按权重降序排序
    results.sort(key=lambda x: x[1].weight, reverse=True)

    # 更新缓存（如果启用且达到阈值）
    if CACHED and top_node.search_count >= CACHED_THREHOLD:
        top_node.cache = results[:CACHED_SIZE]

    return results[:limit] if limit else results

# add 函数：用于向树中添加关键词。它逐字符遍历关键词，逐层构建节点。如果当前字符不存在于当前层，则创建一个新节点。如果是关键词的最后一个字符，则标记为叶子节点，并设置权重。
def add(node: Node, keyword: str, weight=0, **kwargs):
    """
    添加关键词到树中，带有权重（热度）
    """
    current: Node = node
    for i, char in enumerate(keyword):
        if char not in current:
            # 如果是最后一个字符，标记为叶子节点
            is_leaf = (i == len(keyword) - 1)
            new_node = Node(char, is_leaf=is_leaf, weight=weight if is_leaf else 0, **kwargs)
            current.add_subnode(new_node)
            current = new_node
        else:
            current = current.get_subnode(char)
        # 清除受影响路径的缓存
        if CACHED:
            current.cache = []
    # 确保最后一个节点属性正确
    current.is_leaf = True
    current.weight = weight
    for k, v in kwargs.items():
        setattr(current, k, v)


def delete(node, keyword, judge_leaf=False):
    """
    从树中删除关键词
    """
    if not keyword:
        return
    top_node = node.get_top_node(keyword)
    if top_node is None:
        return
    # 清除缓存
    if CACHED:
        top_node.cache = []

    if judge_leaf:
        if top_node.is_leaf:
            return
    elif not top_node.is_leaf:
        return

    if top_node.has_subnode():
        top_node.is_leaf = False
    else:
        # 删除空节点
        parent_prefix = keyword[:-1]
        parent_node = node.get_top_node(parent_prefix)
        if parent_node:
            del parent_node[top_node.key]
            delete(node, parent_prefix, judge_leaf=True)

# build 函数：通过输入的关键词列表（data_list），调用 add 函数构建整棵树。
def build(data_list):
    """
    构建带权重的树
    @param data_list: [(keyword, weight), ...] 的列表
    """
    root = Node("")
    for keyword, weight in data_list:
        if isinstance(keyword, str):  # 过滤非字符串
            add(root, keyword, weight)
    return root


# 使用示例
# if __name__ == "__main__":
    # 测试数据 (关键词, 热度)，输入数据格式如下：
# test_data = [
#     ("apple", 100),
#     ("app", 50),
#     ("application", 80),
#     ("banana", 60),
#     ("band", 70)
# ]

# 构建树
# tree = build(test_data)

# 搜索测试
# print("Search 'pp':")
# for word, node in search(tree, "app"):
#     print(f"{word}: {node.weight}")

# print("\nSearch 'ba':")
# for word, node in search(tree, "ba"):
#     print(f"{word}: {node.weight}")