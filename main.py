from graphviz import Digraph

# 创建一个有向图
er = Digraph('ER Diagram', filename='er_diagram', format='png')
er.attr(rankdir='LR', size='8,5')

# 账号表 accounts
er.node('accounts', '''{
    accounts |
    + username : string \l
    + password : string \l
}''', shape='record')

# 用户信息表 user_info
er.node('user_info', '''{
    user_info |
    + username : string \l
    + used_storage_mb : float \l
    + points : int \l
    + record_id : string \l
}''', shape='record')

# 分类记录表 records
er.node('records', '''{
    records |
    + record_id : string \l
    + username : string \l
    + type : int \l
    + storage_used_mb : float \l
    + points_used : int \l
    + timestamp : datetime \l
    + image_count : int \l
}''', shape='record')

# 关系（外键）
er.edge('accounts', 'user_info', label='username')
er.edge('user_info', 'records', label='record_id')
er.edge('accounts', 'records', label='username')

er.render('/mnt/data/er_diagram', view=False)
'/mnt/data/er_diagram.png'
