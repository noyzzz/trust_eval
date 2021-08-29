from collections import deque
import scipy.io


def get_trust_array():
    trust = scipy.io.loadmat('trustnetwork.mat')
    return trust['trustnetwork']


def get_rating_array():
    rating = scipy.io.loadmat('rating.mat')
    return rating['rating']


def get_filtered_rating_array(num_rows=-1):
    raw_ratings = get_rating_array()
    return raw_ratings[raw_ratings[:, 2].argsort()][0:num_rows]


def get_filtered_trust_array():
    filtered_rating_array = get_filtered_rating_array()
    authors_raw = (filtered_rating_array[:, 0])
    authors_set = set(authors_raw)
    trust_array_raw = get_trust_array()
    trust_arr_filtered = []
    for trust_row in trust_array_raw:
        if trust_row[0] in authors_set and trust_row[1] in authors_set:
            trust_arr_filtered.append(trust_row)
    return trust_arr_filtered


def get_nodes_id_set(trust_array):
    nodes_list = []
    for trust_row in trust_array:
        nodes_list.append(trust_row[0])
        nodes_list.append(trust_row[1])
    return set(nodes_list)


def make_graph(trust_array):
    nodes_ids = get_nodes_id_set(trust_array)
    nodes_dict = {}
    for node_id in nodes_ids:
        nodes_dict[node_id] = Node(node_id)
    for trust_row in trust_array:
        truster = trust_row[0]
        trustee = trust_row[1]
        nodes_dict[truster].all_neighbors.append(nodes_dict[trustee])
    return list(nodes_dict.values())


class Node():
    def __init__(self, id):
        self.id = id
        self.all_neighbors = []
        self.local_neighbors = []
        self.longer_contacts = []
        self.longest_contacts = []
        self.active_domain = set()


def CBFS(nodes, src, sink):
    """
    :param nodes: nodes should have sorted lists as neighbors
    :param src: truster
    :param sink: trustee
    :return: list of paths from src to sink
    """
    visited = {}
    queue = deque()
    par = {}
    nodes_dict = {}
    queue.append((-1, src.id))
    paths = []
    for node in nodes:
        nodes_dict[node.id] = node
    while (len(queue) > 0):
        (current_node_parent_id, current_node_id) = queue.popleft()
        if current_node_id == sink.id:
            paths.append((current_node_parent_id, current_node_id))
            continue
        if current_node_id in visited:
            continue
        visited[current_node_id] = 1
        par[current_node_id] = current_node_parent_id
        added_neighbor_count = 0
        neighbor_level = 0
        current_node = nodes_dict[current_node_id]
        listed_neighbors_count = len(current_node.longest_contacts) + len(current_node.longer_contacts) + len(
            current_node.local_neighbors)
        while added_neighbor_count < listed_neighbors_count:
            if neighbor_level < len(current_node.longest_contacts):
                queue.append((current_node_id, current_node.longest_contacts[neighbor_level].id))
                added_neighbor_count += 1
            if neighbor_level < len(current_node.longer_contacts):
                queue.append((current_node_id, current_node.longer_contacts[neighbor_level].id))
                added_neighbor_count += 1
            if neighbor_level < len(current_node.local_neighbors):
                queue.append((current_node_id, current_node.local_neighbors[neighbor_level].id))
                added_neighbor_count += 1
            neighbor_level += 1
    return par, paths


def users_active_domain_filler(nodes, rating_array):
    nodes_dict = {}
    for node in nodes:
        nodes_dict[node.id] = node
    for rating_node in rating_array:
        if rating_node[0] not in nodes_dict: continue
        nodes_dict[rating_node[0]].active_domain.add(rating_node[2])
        # if(len(nodes_dict[rating_node[0]].active_domain) > 3):
        #     print(str(rating_node[0]) + "  "  + str(nodes_dict[rating_node[0]].active_domain))


def main_test1():
    nodes = []
    for i in range(11):
        nodes.append(Node(i))
    nodes[0].all_neighbors.append(nodes[1])
    nodes[0].all_neighbors.append(nodes[2])
    nodes[0].all_neighbors.append(nodes[3])
    nodes[0].all_neighbors.append(nodes[4])
    nodes[0].longest_contacts.append(nodes[1])
    nodes[0].longest_contacts.append(nodes[2])
    nodes[0].longest_contacts.append(nodes[3])
    nodes[0].longest_contacts.append(nodes[4])

    nodes[1].all_neighbors.append(nodes[10])
    nodes[1].longest_contacts.append(nodes[10])

    nodes[4].all_neighbors.append(nodes[10])
    nodes[4].longest_contacts.append(nodes[10])

    nodes[2].all_neighbors.append(nodes[5])
    nodes[2].all_neighbors.append(nodes[6])
    nodes[2].longer_contacts.append(nodes[5])
    nodes[2].longest_contacts.append(nodes[6])

    nodes[3].all_neighbors.append(nodes[7])
    nodes[3].all_neighbors.append(nodes[8])
    nodes[3].longest_contacts.append(nodes[7])
    nodes[3].longest_contacts.append(nodes[8])

    nodes[5].all_neighbors.append(nodes[9])
    nodes[5].longest_contacts.append(nodes[9])

    nodes[6].all_neighbors.append(nodes[9])
    nodes[6].longest_contacts.append(nodes[9])

    nodes[7].all_neighbors.append(nodes[10])
    nodes[7].longest_contacts.append(nodes[10])

    nodes[9].all_neighbors.append(nodes[10])
    nodes[9].longest_contacts.append(nodes[10])

    (par, paths) = CBFS(nodes, nodes[0], nodes[10])
    #print(paths)


def compute_priority(nodes_dict, target, neighbor, topic_domain):
    LAMBDA1 = 1.0
    LAMBDA2 = 1.0
    xj = 1 if topic_domain in nodes_dict[neighbor].active_domain else 0
    yj = len(nodes_dict[target].active_domain.intersection(nodes_dict[neighbor].active_domain))
    return ((LAMBDA1 * xj + LAMBDA2 * yj) / (1 + len(nodes_dict[target].active_domain)))


def compute_social_distance(nodes_dict, src, neighbor):
    return len(nodes_dict[neighbor].active_domain) \
           - len(nodes_dict[src].active_domain.intersection(nodes_dict[neighbor].active_domain)) + 1


def split_neighbors_in_lists(nodes_dict, node_id):
    this_node = nodes_dict[node_id]
    if len(this_node.local_neighbors) > 0 or len(this_node.longer_contacts) > 0 or len(this_node.longest_contacts) > 0:
        raise Exception("neighbor lists should be empty")
    for neighbor in this_node.all_neighbors:
        social_dis = compute_social_distance(nodes_dict, this_node.id, neighbor.id)
        if social_dis == 1:
            this_node.local_neighbors.append(neighbor)
        elif social_dis > 1 and social_dis < len(neighbor.active_domain):
            this_node.longer_contacts.append(neighbor)
        elif social_dis > 1 and social_dis == len(neighbor.active_domain):
            this_node.longest_contacts.append(neighbor)


def sort_nodes_neighbors(nodes_dict, topic_domain, target_id):
    for node in list(nodes_dict.values()):
        node.local_neighbors.sort(key=lambda x: compute_priority(nodes_dict, target_id, x.id, topic_domain),
                                  reverse=True)
        node.longest_contacts.sort(key=lambda x: compute_priority(nodes_dict, target_id, x.id, topic_domain),
                                   reverse=True)
        node.longer_contacts.sort(key=lambda x: compute_priority(nodes_dict, target_id, x.id, topic_domain),
                                  reverse=True)


def generate_full_paths(par, paths):
    out_list = []
    for this_path_tuple in paths:
        this_list = []
        last_src = this_path_tuple[0]
        dst = this_path_tuple[1]
        this_list.insert(0,dst)
        cur = last_src
        while cur != -1:
            this_list.insert(0, cur)
            cur = par[cur]
        out_list.append(this_list)
    return out_list


def main_test2():
    trust_array = get_filtered_trust_array()
    rating_array = get_filtered_rating_array()
    nodes = make_graph(trust_array)
    #print(type(nodes))
    users_active_domain_filler(nodes, rating_array)
    nodes_dict = {}
    for node in nodes:
        nodes_dict[node.id] = node
    for node in nodes:
        split_neighbors_in_lists(nodes_dict, node.id)
    # for node1 in nodes:
    #     for node2 in nodes:
    #         for cat_id in range(28):
    #             if node1 == node2: continue
    #             sort_nodes_neighbors(nodes_dict,cat_id,node2.id)
    #             (par, paths) = CBFS(nodes,node1,node2)
    #             if len(paths) > 0 :
    #                 print(
    #                     "node1 : {0} node2 :  {1} cat_id :  {2} path_length  : {3}".format(str(node1.id), str(node2.id),
    #                                                                                        str(cat_id),
    #                                                                                        str(len(paths))))
    TH = 0.5
    DOMAIN_CODE_ = 8
    SRC_NODE =  15373
    DST_NODE = 9831
    for DOMAIN_CODE in nodes_dict[SRC_NODE].active_domain:
        sort_nodes_neighbors(nodes_dict, DOMAIN_CODE, DST_NODE)
        (par, paths) = CBFS(nodes, nodes_dict[SRC_NODE], nodes_dict[DST_NODE])
        full_paths = generate_full_paths(par, paths)
        approved_full_paths = []
        for full_path in full_paths:
            full_path_approved = True
            for i in range(len(full_path)-1):
                edge_src = full_path[i]
                edge_dst = full_path[i+1]
                edge_priority = compute_priority(nodes_dict,edge_dst, edge_src, DOMAIN_CODE)
                #print(str(edge_priority) + '\n')
                if edge_priority < TH:
                    full_path_approved = False
                    break
            if full_path_approved:
                approved_full_paths.append(full_path)
        print(str(DOMAIN_CODE) + "  " + str(len(approved_full_paths)) + "  " + str(len(full_paths)))

        if(len(approved_full_paths) < 5):
            for path_app in approved_full_paths:
                print(path_app)
        print()
        print()
    #print(paths)


main_test2()
