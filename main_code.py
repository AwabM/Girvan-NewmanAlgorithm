
import time, os, sys, json, csv, collections, itertools, operator, math, random , collections, copy
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


##########################################################
#######    Community Detection Functions       ###########
def check_div_by_zero(x):
    if x[1][1] == 0:
        return (x[0], 0.0)
    else:
        return (x[0], x[1][0] / x[1][1])

def limit_the_rating(x):
    if x[1] < 0:
        return (x[0], 0.0)
    else:
        if x[1] > 5:
            return (x[0], 5.0)
        else:
            return x

def reshape_after_join_urs(x):
    a = x[0]
    baa = x[1][0][0]
    bab = x[1][1][0]

    ba = baa - bab
    bba = x[1][0][1]

    bbb = x[1][1][1]

    bb =  bba - bbb
    return (a,(ba, bb))

def complete_weight_formela(x):
    dom = math.sqrt(x[1][0] * math.sqrt(x[1][1]))
    return (x[0], dom)

def reshape_after_join_train_test(x):
    training_user = x[1][1][0]
    test_business = x[0]
    test_user = x[1][0]
    training_rate = x[1][1][1]
    return ( training_user , (test_user,test_business, training_rate))

# ( training_user , ( (test_user,test_business, training_rate), (training_user_sum, training_user_size) ) )
def reshape_after_join_train_test_with_avr(x):
    training_user_sum = x[1][1][0]
    training_user_size = x[1][1][1]
    training_rate = x[1][0][2]
    a = (training_user_size - 1)
    if a == 0:
        final = 0
    else:
        final = (training_user_sum - training_rate)/ a
    test_user = x[1][0][0]
    training_user = x[0]
    test_business = x[1][0][1]
    training_rate = x[1][0][2]
    return ((test_user,training_user), (test_business, training_rate - final))

def reshape_Pearson_right(x):
    return (x[0][0], (x[0][1], x[1]))

def reshape_Pearson(x):
    left = x[1][0]
    right = x[1][1][1]
    return ((x[0], x[1][1][0]), left + right)

def reorder_key(x):
    key = tuple(sorted(x[0]))
    return (key, x[1])


###########

def find_jaccard_sim(pair, characteristic_mat):
    p_one = set(characteristic_mat[pair[0]])
    p_two = set(characteristic_mat[pair[1]])
    sim = len(p_one.intersection(p_two))*1.0 / len(p_one.union(p_two))
    return ((pair[0], pair[1]) , sim)

def find_usr_index(inxex_dict, x):
    return (x[1], inxex_dict[x[0]])

# Take one product and find all minHash for this product
def minhashing(bbb, hash_vals):
    minhash_vals = []
    for vals in hash_vals:
        #print("\n  vals, bbb[1] :", vals, bbb[1])
        # ('\n  vals, bbb[1] :', [71153, 'zzo--VpSQh8PpsGVeMC1dQ', 27527], [5699, 7421, 4230, 6791, 3300, 2564, 2838, 9497, 7594, 8895, 8956, 521, 10748])
        # ('\n b[1], vals[0] :', [37571, 28099, 46489], [5699, 7421, 4230, 6791, 3300, 2564, 2838, 9497, 7594, 8895, 8956, 521, 10748])
        # ('\n b[1], vals[0] :', 39569, [5699, 7421, 4230, 6791, 3300, 2564, 2838, 9497, 7594, 8895, 8956, 521, 10748])
        minhash_vals.append(min([((vals[0] * usr_index  +  vals[1]) % vals[2]) % m for usr_index in bbb[1]]))
    return (bbb[0], minhash_vals)


def emit_pairs(list_of_products):
    product_pairs = sorted(list(itertools.combinations(list(list_of_products) , 2)))
    return [(i,1) for i in product_pairs]

##user is (user_id, [(business_id, avrRate), (business2_id, avrRate), .. ] )
## userWeightedRatingGroupList is list of above for all users
def find_weight_for_business(curr_bus, curr_bus_rates, other_bus_rating_list):
    curr_buss_weights_dic = {}
    busRatingDict = dict(curr_bus_rates) # userRatingDict
    dominator1 = sum([j * j for j in busRatingDict.values()])
    weightesDict = {}  ## dict for all other bus with weight w(bus, other_user)
    # print("\n\n\n From findPAirsWeights: ", userWeightedRatingGroupList[1])
    for i in range(len(other_bus_rating_list)):
        other_bus_id = other_bus_rating_list[i][0]
        other_bus_rates =  dict(other_bus_rating_list[i][1])
        numorator = 0
        dominator2 = sum([j * j for j in other_bus_rates.values()])
        dominator = dominator1 * dominator2
        for k in  set(busRatingDict.keys()) & set(other_bus_rates.keys()):
            numorator += busRatingDict[k] * other_bus_rates[k]

        if numorator == 0 or dominator == 0 :
            continue
        else:
            curr_buss_weights_dic[other_bus_id] = numorator / dominator
    # print("\n User and weights:", user[0],weightesDict)
    #if len(curr_buss_weights_dic) != 0:
        #print(curr_bus, curr_buss_weights_dic)
    return (curr_bus, curr_buss_weights_dic)



#######  END Community Detection Functions    ############
##########################################################


##########################################################
###########    Betweenness Functions       ###############
def str_to_int(x, usr_dict , bus_dict):
    user_int = usr_dict[x[0]]
    bus_int = bus_dict[x[1]]
    #print("from int to str: x ", x)
    #print("from int to str: new:", user_int, bus_int, x[2])
    return (usr_dict[x[0]], bus_dict[x[1]], x[2] )



def Girvan(root, f_edgeToEdgesDict):
    i = 0
    top = root
    bfs = [top]
    nodes = set(bfs)
    level_of_node = {top: 0}
    parentsDict = {}

    while i < len(bfs):
        curr_top = bfs[i]
        children = f_edgeToEdgesDict[curr_top]
        for child in children:
            if child not in nodes:
                bfs.append(child)
                nodes.add(child)
                level_of_node[child] = level_of_node[curr_top] + 1
                if curr_top == root:
                    parentsDict[child] = {curr_top:1}
                else:
                    #print("HI   : ",parentsDict[curr_top])
                    parentsDict[child] = {curr_top: sum(parentsDict[curr_top].values())}
            else:
                if level_of_node[child] == level_of_node[curr_top] + 1:
                    parentsDict[child][curr_top] = sum(parentsDict[curr_top].values())
        i += 1

    weights_of_vertexsDict = {}
    for_range = len(bfs)
    for i in range(for_range - 1, -1, -1):
        vertex = bfs[i]
        if vertex not in weights_of_vertexsDict:
            weights_of_vertexsDict[vertex] = 1
        if vertex in parentsDict:
            OneParentsDict = parentsDict[vertex]
            #print("Error:    ",  OneParentsDict )
            parents_nums = sum(OneParentsDict.values())
            for parent in OneParentsDict.keys():
                weight = (float(weights_of_vertexsDict[vertex]) / parents_nums) * OneParentsDict[parent]

                if parent not in weights_of_vertexsDict:
                    weights_of_vertexsDict[parent] = 1
                weights_of_vertexsDict[parent] += weight
                yield (tuple(sorted([vertex, parent])), weight / 2)

############   END Betweenness Functions    ##############
##########################################################


##########################################################
###########    Modularity Functions       ###############

############# Fining Modularity for each cut ####################
## delete duplicates edges ( src-dist && dist-src )
def removeEdge(o_dictOfEdges, delEdge ):
    dictOfEdges = copy.deepcopy(o_dictOfEdges)
    src = delEdge[0]
    dist =  delEdge[1]
    f_not_deleted1 = True
    f_not_deleted2 = True
    if src in dictOfEdges:
        if dist in set(dictOfEdges[src]):

            dictOfEdges[src].remove(dist)
            f_not_deleted1 = False
    if dist in dictOfEdges:
        if src in set(dictOfEdges[dist]):
            dictOfEdges[dist].remove(src)
            f_not_deleted2 = False
    if f_not_deleted1 and f_not_deleted2:
        print(" deleted only one or none: ", delEdge )
        exit()

    if len(dictOfEdges[src]) == 0: del dictOfEdges[src]
    if len(dictOfEdges[dist]) == 0: del dictOfEdges[dist]
    return dictOfEdges

def find_new_communites(f_communites, f_current_edgesDict, f_deleted_edge ):
    split_list = []
    new_comunites = []
    sl_index = 'null'
    for comm_index in range(len(f_communites)):
        if len(set(f_deleted_edge) - set(f_communites[comm_index])) == 0:
            split_list = copy.deepcopy(f_communites[comm_index])
            sl_index = comm_index
        else:
            new_comunites.append(f_communites[comm_index])
    if not split_list:
        print(" ERROr Can't find EDGE")
        exit()

    comm1 = []
    need_to_visit = [f_deleted_edge[0]]
    i2 = 0
    while need_to_visit:
        i2+=1
        next_head = need_to_visit.pop(0)
        if not next_head in set(comm1):
            comm1.append(next_head)
            if next_head in f_current_edgesDict.keys():
                children = f_current_edgesDict[next_head]
                for child in children:
                    need_to_visit.append(child)

    comm2 = []
    need_to_visit = [f_deleted_edge[1]]
    i1 = 0
    while need_to_visit:
        i1 +=1
        next_head = need_to_visit.pop(0)
        if not next_head in set(comm2):
            comm2.append(next_head)
            if next_head in f_current_edgesDict.keys():
                children = f_current_edgesDict[next_head]
                for child in children:
                    need_to_visit.append(child)
    if not (set(comm1) | set([f_deleted_edge[1]])) - (set(comm2) | set(f_deleted_edge[0])):
        new_comunites.append(split_list)
        return new_comunites , False
    else:
        if not set(comm1) & set(comm2):
            new_comunites.append(comm1)
            new_comunites.append(comm2)
            return new_comunites, True
    print(" issue No full connected nor sperated ????")
    exit()

def find_Modararity(f_comm, f_m, f_current_edgesDict , f_edgesDegrees):
    com_id = f_comm[0]
    comm_members = f_comm[1]
    if len(set(comm_members)) != len(comm_members):
        print(" More than one same member in compunity",   f_comm  )
        exit()
    comm_pairs = list(itertools.combinations(comm_members, 2))
    for f_pair in comm_pairs:
        if f_pair[0] in f_current_edgesDict.keys() and f_pair[1] in set(f_current_edgesDict[f_pair[0]]):
            f_expected = 1 - ( (f_edgesDegrees[f_pair[0]] * f_edgesDegrees[f_pair[1]]) / (2 * f_m) )
            to_out = ((com_id,f_m), f_expected )
            yield to_out
        else:
            f_expected = 0 - ( (f_edgesDegrees[f_pair[0]] * f_edgesDegrees[f_pair[1]]) / (2 * f_m) )
            to_out = ((com_id,f_m) , f_expected)
            yield to_out
            if f_pair[1] in f_current_edgesDict.keys() and f_pair[0] in set(f_current_edgesDict[f_pair[1]]):
                print("Directed edege is found !!!")
                exit()

############   END Modularity Functions    ##############
##########################################################



# Setting the envirnment
sc = SparkContext('local[*]', 'GN-Algo')
train_f_path = sys.argv[1]
betweenness_output_f_path = sys.argv[2]
community_output_f_path = sys.argv[3]


############   Define variables    ##############
# customized partitioning: 8 is enough for personal computers. If you run this code on a
#server, you might consider using defult partition to get better performance.
num_partition = 8
m = 0 # used inside some Functions
start_time = time.time()
edgeToEdgesDict = {}
############   END of Define variables    ##############


############# START of Part One  ####################
train_rdd = sc.textFile(train_f_path)
train_first_line = train_rdd.first()
train_rdd = train_rdd.map(lambda x: x.split(' ')) #.partitionBy(num_partition, partition_f)

# Finding all uniqe vertices
train_rddList = train_rdd.collect()
i = 0
for e in train_rddList:
    i += 1
    #print("e0 e1 are: ", i,  e[0], e[1] )
    if e[0] in edgeToEdgesDict.keys():
        edgeToEdgesDict[e[0]].append(e[1])
    else:
        edgeToEdgesDict[e[0]] = [e[1]]
    if e[1] in edgeToEdgesDict.keys():
        edgeToEdgesDict[e[1]].append(e[0])
    else:
        edgeToEdgesDict[e[1]] = [e[0]]


# Create the structure of GN algo
vertices_left = train_rdd.map(lambda x: (x[0],1))
vertices_right = train_rdd.map(lambda x: (x[1],1))
vertices = vertices_left.union(vertices_right).reduceByKey(lambda x, y: x + y).map(lambda x: x[0])


betweennessRDD = vertices.flatMap(lambda x: Girvan(x, edgeToEdgesDict )).reduceByKey(lambda x, y: x + y)
betweennessDict = dict(betweennessRDD.collect())


############# Prepare Writing Betweenness file ####################
train_dataList = train_rdd.collect()
betweennessDictForWriting = collections.defaultdict(list)
for i in train_dataList:
    tuple_i = tuple(i)
    tuple_i_revers = (tuple_i[1], tuple_i[0])
    if tuple_i in betweennessDict.keys():
        line = tuple(sorted(tuple_i))
        betweennessDictForWriting[betweennessDict[tuple_i]].append(line)
    else:
        if tuple_i_revers in betweennessDict.keys():
            line = tuple(sorted(tuple_i))
            betweennessDictForWriting[betweennessDict[tuple_i_revers]].append(line)
        else:
            line = tuple(sorted(tuple_i))
            betweennessDictForWriting[betweennessDict[tuple_i_revers]].append(line)
############# END Prepare Writing Betweenness file ####################


############# Writing to Betweenness file  ####################
mb_out = open(betweenness_output_f_path, 'a+')
#mb_out.write("user_id, business_id, actual , prediction:\n")
for i in sorted(betweennessDictForWriting.keys(), reverse=True):
    i_list = betweennessDictForWriting[i]
    #print(i_list)
    for tuple_ in sorted(i_list, key=lambda e: str(e)):
        mb_out.write(str(tuple_) + ', ' + str(i))
        mb_out.write('\n')
mb_out.close()
############# END Writing to Betweenness file ####################


############# END of Part One  ####################



############# START of Part Two  ####################
"""
Part two implemented in 5 steps:
While there is still some edges to delete
1: Delete highest edge in G
2: compute Modularity for G’
3: report edges deleted so far + Modularity
4: compute betwenness in G’
5: go back to line1
"""

###### Find ki kj for each vertex ###
edgesDegrees = dict([(key, len(edgeToEdgesDict[key]) ) for key in edgeToEdgesDict.keys()])
###### End of fining ki kj #####

############   Define variables    ##############
cut_options_and_mods = []
current_edgesDict = edgeToEdgesDict
curr_betweennessDict = betweennessDict
curr_edgesDegrees = edgesDegrees
deleted_edges_so_far = []
communitesDict  = []
############   END of Define variables    ##############

communitesDict.append([vertices.collect()])
cut_options_and_mods.append((0, []))

vers_for_mod = vertices.collect()
inside_formula = {}
main_graph_mod = 0
for i in vers_for_mod:
    for j in vers_for_mod:
        sort_tupl = tuple(sorted((i, j)))
        if sort_tupl not in inside_formula.keys():
            if j in set(edgeToEdgesDict[i]):
                aij = 1.0
            else:
                if i in set(edgeToEdgesDict[j]):
                    print(" Directed Link")
                    break
                aij = 0.0
            inside_formula[sort_tupl] = aij - (( len(edgeToEdgesDict[i]) * len(edgeToEdgesDict[j]) ) / 1812.0)
            main_graph_mod += inside_formula[sort_tupl]


def simpleModularity(f_comm, f_inside_formula):
    visited_so_far = set()
    total_mod = 0
    for i in f_comm:
        for j in f_comm:
            sort_tupl = tuple(sorted((i,j)))
            if sort_tupl not in visited_so_far:
                total_mod += f_inside_formula[sort_tupl]
                visited_so_far.add(sort_tupl)
    return total_mod


m = 906 # totol num of edges in the network
i_through_while = 0
best_modalurity = -1
best_comm = [vertices.collect()]
boost = 5
while (current_edgesDict):
    i_through_while += 1
    find_new_com = False
    while not find_new_com:
        highest_edge = max(curr_betweennessDict.items(), key=operator.itemgetter(1))[0]
        deleted_edges_so_far.append(highest_edge)
        del curr_betweennessDict[highest_edge]
        current_edgesDict = removeEdge(current_edgesDict, highest_edge)
        # curr_edgesDegrees = dict([(key, len(current_edgesDict[key])) for key in current_edgesDict.keys()])
        potintial_best_comm, find_new_com  = find_new_communites(communitesDict[-1], current_edgesDict, highest_edge)
        #print(" \n\n\n\n\n\n Len BEFORE and AFTER :::::::                              $$#$#$#$##$##$#@#$#$#$#$##$#$#", len(communitesDict[-1]), len(potintial_best_comm))
        communitesDict.append(potintial_best_comm)
        curr_betweennessDict = dict(vertices.flatMap(lambda x: Girvan(x, current_edgesDict)).reduceByKey(lambda x, y: x + y).collect())
    #print("  1- Number of cupunites now, edges cutted ", len(communitesDict[-1]), len(deleted_edges_so_far))
    communitesWithIds = list(enumerate(communitesDict[-1]))
    # [(0, ['a', 'b', 'bb']), (1, ['c', 'd', 'dd']), (2, ['e', 'f', 'ff'])]
    #print("2 - communitesWithIds: ", communitesWithIds)
    ### NEW
    total_mod = 0
    for i in potintial_best_comm:
        total_mod += simpleModularity(i ,inside_formula)
    this_loop_modularity = total_mod / 1812

    if this_loop_modularity > best_modalurity:
        best_modalurity = this_loop_modularity
        best_comm = potintial_best_comm
    ## End New
    cut_options_and_mods.append((this_loop_modularity, copy.deepcopy(deleted_edges_so_far), i_through_while))
    #print(" 3- last Mod and compunites is/ with # of cuts: ", cut_options_and_mods[-1][0],
          #len(cut_options_and_mods[-1][1]))
    #print(" 4 - Number of above compunites are: ", len(communitesWithIds))

    #print("  :::     Done    ")
    #if this_loop_modularity == 0.49175260451782016:
    #    print("Top at: ", communitesDict[len(deleted_edges_so_far)] )
    #    exit()
    #if i_through_while > 100:
    #    break
    ## Find new betweeness edges
    if i_through_while > 25:
        #print(" BREAK because I found 19       iwhile is:", i_through_while)
        break
    m -= 1


best_cut = sorted(cut_options_and_mods, reverse=True )[0]


############# Writing to community file  ####################
mb_out = open(community_output_f_path, 'a+')

## Sort the result based on the first vertex
sortedDict = collections.defaultdict(list)
for i in best_comm:
    sortedDict[len(i)].append(i)

for i in sorted(sortedDict.keys()):
    members = sorted(sortedDict[i])
    sorted_members = []
    for j in members:
        sorted_members.append(sorted(j))
    for m in sorted(sorted_members):
        s = sorted(m)
        mb_out.write( str(s)[1:-1])
        mb_out.write('\n')
mb_out.close()
############# END Writing to community file ####################

############# END of Part Two  ####################


print("Duration: ", time.time() - start_time)
exit()
