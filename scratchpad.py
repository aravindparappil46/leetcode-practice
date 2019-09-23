
deps_dict = {
    "factual-commons" : ['apache-commons', 'guava', 'thrift'],
    "map-reduce" : ['apache-commons', 'hadoop'],
    "place-attach" : ['factual-commons', "map-reduce"],
    'hive': ['hadoop', 'apache-commons'],
    'hive-querier': ['hive', 'factual-commons']
}

a = [1,2,4,5,6]

b =[111]

b += [i for i in a]
print(list(b))