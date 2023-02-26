import numpy as np
from numpy import float64 as double, ndarray as array, int32 as long
from numba import njit
from numba.typed import List
from env import getStateSize, getActionSize


@njit
def init_tree(F0: array, left_arr: array, right_arr: array, max_node: int):
    if max_node < 1:
        raise Exception("Không khởi tạo được cây khi max_node < 1.")

    stateSize = getStateSize()
    actionSize = getActionSize()

    # Attribute arrays
    attributes = np.zeros((max_node, stateSize), double)
    attributes[0] = F0

    # Bias arrays
    biases = np.zeros((max_node+1, actionSize), double)
    biases[0] = left_arr
    biases[1] = right_arr

    # Node informations
    '''
    Một node information array có index là i trong "node informations",
    chứa thông tin về node quyết định có cùng index trong "attributes":
    * [0] - xác định left_child:
        + Nếu value là chẵn thì left_child là một node quyết định khác,
        node được link đến có index = value // 2.
        + Nếu value là lẻ thì left_child là một node kết quả,
        bias array được link đến có index = value // 2 (= (value - 1) / 2).
    * [1] - xác định right_child, tương tự như left_child.
    * [2] - xác định node cha.
        + Đối với node quyết định gốc (root) thì value = -1
        + Đối với node quyết định khác, value = index của node cha.
    '''
    node_informations = np.zeros((max_node, 3), double)
    node_informations[0] = np.array([1, 3, -1], double)

    # Check các vị trí của các attribute array
    infor_att = np.zeros((1, max_node), double)
    infor_att[0][0] = 1

    # Check các bị trí của các bias array
    infor_bias = np.zeros((1, max_node+1), double)
    infor_bias[0][0:2] = 1

    #
    tree = List()
    tree.append(attributes)
    tree.append(biases)
    tree.append(node_informations)
    tree.append(infor_att)
    tree.append(infor_bias)

    return tree


@njit
def insert_after(tree: List, node_idx: int, F: array, parent_branch: bool, child_branch: bool, bias_arr: array):
    '''
    Chèn vào sau node quyết định có index là node_idx:
    * F: thuộc tính của node mới.
    * parent_branch: Node mới sẽ được chèn vào nhánh nào của node cha.
    * child_branch: Nhánh con cũ của node cha sẽ được gắn vào nhánh nào của node mới.
    * bias_arr: Bias array ở nhánh còn lại của node mới.

    Hàm sẽ trả ra các giá trị để báo lỗi:
    * 0: Không có lỗi, chèn thành công.
    * 1: Ở index "node_idx" chưa được tạo node, do đó không xác định được node cha.
    * 2: Số node quyết định đã đạt tối đa, không thể chèn thêm node.
    '''
    attributes = tree[0]
    biases = tree[1]
    node_informations = tree[2]
    infor_att = tree[3][0]
    infor_bias = tree[4][0]
    parent_branch = long(parent_branch)
    child_branch = long(child_branch)

    if infor_att[node_idx] == 0 or node_idx < 0:
        return 1

    temp = np.where(infor_att == 0)[0]
    if temp.shape[0] == 0:
        return 2

    # Thêm các thuộc tính và bias array cho node mới.
    new_node_idx = temp[0]
    attributes[new_node_idx] = F
    infor_att[new_node_idx] = 1

    new_bias_idx = np.where(infor_bias == 0)[0][0]
    biases[new_bias_idx] = bias_arr
    infor_bias[new_bias_idx] = 1

    # Link node mới và các node liên quan
    parent_node_infor = node_informations[node_idx]
    new_node_infor = node_informations[new_node_idx]

    new_node_infor[1-child_branch] = 2*new_bias_idx + 1
    new_node_infor[child_branch] = parent_node_infor[parent_branch]
    new_node_infor[2] = node_idx
    parent_node_infor[parent_branch] = 2*new_node_idx

    if new_node_infor[child_branch] % 2 == 0:
        child_node_idx = long(new_node_infor[child_branch] // 2)
        node_informations[child_node_idx][2] = new_node_idx

    return 0


@njit
def __delete_node(tree: List, node_idx: int):
    '''
    Xóa node có index là node_idx cùng với tất cả nhánh con của nó.
    '''
    attributes = tree[0]
    biases = tree[1]
    node_informations = tree[2]
    infor_att = tree[3][0]
    infor_bias = tree[4][0]

    node_infor = node_informations[node_idx]
    for i in range(2):
        if node_infor[i] % 2 == 1: # Bias array
            bias_idx = long(node_infor[i] // 2)
            biases[bias_idx][:] = 0
            infor_bias[bias_idx] = 0
        else: # Node khác
            _node_idx = long(node_infor[i] // 2)
            __delete_node(tree, _node_idx)

    attributes[node_idx][:] = 0
    infor_att[node_idx] = 0
    node_infor[:] = 0
    return


@njit
def insert_after_and_replace_child_branch(tree: List, node_idx: int, F: array, parent_branch: bool, left_arr: array, right_arr: array):
    '''
    Chèn vào sau node quyết định có index là node_idx:
    * F: Thuộc tính của node mới.
    * parent_branch: Node mới sẽ được chèn vào nhánh nào của node cha.
    * left_arr: Nhánh False của node mới.
    * right_arr: Nhánh True của node mới.

    Nhánh con cũ của node cha sẽ được thay thế.

    Hàm sẽ trả ra các giá trị để báo lỗi:
    * 0: Không có lỗi, chèn thành công.
    * 1: Ở index "node_idx" chưa được tạo node, do đó không xác định được node cha.
    * 2: Số node quyết định đã đạt tối đa, không thể chèn thêm node.
    '''
    biases = tree[1]
    node_informations = tree[2]
    infor_att = tree[3][0]
    infor_bias = tree[4][0]
    parent_branch = long(parent_branch)

    if infor_att[node_idx] == 0 or node_idx < 0:
        return 1

    parent_node_infor = node_informations[node_idx]
    p_branch = parent_node_infor[parent_branch]
    if p_branch % 2 == 1: # Bias array
        temp = np.where(infor_att == 0)[0]
        if temp.shape[0] == 0:
            return 2

        bias_idx = long(p_branch // 2)
        biases[bias_idx] = left_arr
        insert_after(tree, node_idx, F, parent_branch, False, right_arr)
    else: # Node khác
        node_del_idx = long(p_branch // 2)
        __delete_node(tree, node_del_idx)
        new_bias_idx = np.where(infor_bias == 0)[0][0]
        biases[new_bias_idx] = left_arr
        infor_bias[new_bias_idx] = 1
        parent_node_infor[parent_branch] = 2*new_bias_idx + 1
        insert_after(tree, node_idx, F, parent_branch, False, right_arr)

    return 0


@njit
def insert_before(tree: List, node_idx: int, F: array, branch: bool, bias_arr: array):
    '''
    Chèn vào trước node quyết định có index là node_idx:
    * F: Thuộc tính của node mới.
    * branch: Node cũ sẽ trở thành nhánh nào của node mới.
    * bias_arr: Bias array ở nhánh còn lại của node mới.

    Hàm sẽ trả ra các giá trị để báo lỗi:
    * 0: Không có lỗi, chèn thành công.
    * 1: Ở index "node_idx" chưa được tạo node, do đó không xác định được node cha.
    * 2: Số node quyết định đã đạt tối đa, không thể chèn thêm node.
    '''
    attributes = tree[0]
    biases = tree[1]
    node_informations = tree[2]
    infor_att = tree[3][0]
    infor_bias = tree[4][0]
    branch = long(branch)

    if infor_att[node_idx] == 0 or node_idx < 0:
        return 1

    temp = np.where(infor_att == 0)[0]
    if temp.shape[0] == 0:
        return 2

    # Thêm các thuộc tính và bias array cho node mới.
    new_node_idx = temp[0]
    attributes[new_node_idx] = F
    infor_att[new_node_idx] = 1

    new_bias_idx = np.where(infor_bias == 0)[0][0]
    biases[new_bias_idx] = bias_arr
    infor_bias[new_bias_idx] = 1

    # Link node mới và các node liên quan
    child_node_infor = node_informations[node_idx]
    new_node_infor = node_informations[new_node_idx]

    new_node_infor[1-branch] = 2*new_bias_idx + 1
    new_node_infor[2] = child_node_infor[2]
    new_node_infor[branch] = 2*node_idx
    child_node_infor[2] = new_node_idx

    if new_node_infor[2] != -1:
        parent_node_idx = long(new_node_infor[2])
        if node_informations[parent_node_idx][0] == 2*node_idx:
            node_informations[parent_node_idx][0] = 2*new_node_idx
        elif node_informations[parent_node_idx][1] == 2*node_idx:
            node_informations[parent_node_idx][1] = 2*new_node_idx
        else:
            raise Exception("Cây sai.")

    return 0


@njit
def insert_before_and_replace_child_branch(tree: List, node_idx: int, F: array, left_arr: array, right_arr: array):
    '''
    Chèn vào trước node quyết định có index là node_idx:
    * F: Thuộc tính của node mới.
    * left_arr: Nhánh False của node mới.
    * right_arr: Nhánh True của node mới.

    Node cũ sẽ được thay thế.

    Hàm sẽ trả ra các giá trị để báo lỗi:
    * 0: Không có lỗi, chèn thành công.
    * 1: Ở index "node_idx" chưa được tạo node, do đó không xác định được node cha.
    '''
    attributes = tree[0]
    biases = tree[1]
    node_informations = tree[2]
    infor_att = tree[3][0]
    infor_bias = tree[4][0]

    if infor_att[node_idx] == 0 or node_idx < 0:
        return 1

    child_node_infor = node_informations[node_idx]
    parent_node_idx = child_node_infor[2]
    __delete_node(tree, node_idx)

    # Thêm các thuộc tính và bias array cho node mới.
    temp = np.where(infor_att == 0)[0]
    new_node_idx = temp[0]
    attributes[new_node_idx] = F
    infor_att[new_node_idx] = 1

    new_left_idx = np.where(infor_bias == 0)[0][0]
    new_right_idx = np.where(infor_bias == 0)[0][1]
    biases[new_left_idx] = left_arr
    biases[new_right_idx] = right_arr
    infor_bias[new_left_idx] = 1
    infor_bias[new_right_idx] = 1

    new_node_infor = node_informations[new_node_idx]
    new_node_infor[0] = 2*new_left_idx + 1
    new_node_infor[1] = 2*new_right_idx + 1
    new_node_infor[2] = parent_node_idx

    if new_node_infor[2] != -1:
        parent_node_idx = long(new_node_infor[2])
        if node_informations[parent_node_idx][0] == 2*node_idx:
            node_informations[parent_node_idx][0] = 2*new_node_idx
        elif node_informations[parent_node_idx][1] == 2*node_idx:
            node_informations[parent_node_idx][1] = 2*new_node_idx
        else:
            raise Exception("Cây sai.")

    return 0


@njit
def check_attribute(state, F):
    if np.sum(state * F) > 1.0:
        return True

    return False


@njit
def get_decision_from_tree(state, tree):
    attributes = tree[0]
    biases = tree[1]
    node_informations = tree[2]

    root_idx = np.where(node_informations == -1)[0]
    if root_idx.shape[0] != 1:
        raise Exception("Cây sai.")

    current_node_idx = root_idx[0]
    for _ in range(tree[0].shape[0]):
        current_node_infor = node_informations[current_node_idx]
        F = attributes[current_node_idx]
        branch = long(check_attribute(state, F))
        current_node_idx = long(current_node_infor[branch])
        if current_node_idx % 2 == 0: # Node khác
            current_node_idx //= 2
        else: # Bias array
            bias_idx = current_node_idx // 2
            return biases[bias_idx]
    else:
        raise Exception("Lỗi các node link vòng tròn.")
