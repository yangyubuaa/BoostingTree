class GBDTreeNode:
    def __init__(self, depth, max_depth, train_data):
        self.depth = depth
        self.train_data = train_data
        self.average = sum([i[1] for i in train_data]) / len(train_data)
        if depth == max_depth:
            if len(train_data) == 1:
                self.split_res = train_data[0][0]
            else:
                self.split_res, left, right = self.node_split(self.train_data)
            self.left = None
            self.right = None
            return
        if len(train_data) == 1:
            self.split_res = train_data[0][0]
            self.left = None
            self.right = None
        else:
            self.split_res, left, right = self.node_split(self.train_data)
            # print(self.split_res, left, right)
            if left:
                self.left = GBDTreeNode(depth + 1, max_depth, left)
            else:
                self.left = None
            if right:
                self.right = GBDTreeNode(depth + 1, max_depth, right)
            else:
                self.right = None

    def node_split(self, last_result):
        c1 = list()
        c2 = list()
        for split_node in range(len(last_result)):
            left = [i[1] for i in last_result[:split_node + 1]]
            sum_left = 0
            for i in left:
                sum_left = sum_left + i
            average_left = sum_left / len(left)
            right = [i[1] for i in last_result[split_node + 1:]]

            try:
                sum_right = 0
                for i in right:
                    sum_right = sum_right + i
                average_right = sum_right / len(right)
                c1.append(average_left)
                c2.append(average_right)
            except ZeroDivisionError:
                c1.append(average_left)
                c2.append(0)
        m = list()
        for i in range(len(last_result)):
            left_data = [i[1] for i in last_result[:i + 1]]
            # print(left_data)
            right_data = [i[1] for i in last_result[i + 1:]]
            # print(right_data)
            sum_left = 0
            for left in left_data:
                sum_left = sum_left + pow((left - c1[i]), 2)
            avg_left = sum_left

            sum_right = 0
            for right in right_data:
                sum_right = sum_right + pow((right - c2[i]), 2)
            avg_right = sum_right
            result = avg_left + avg_right
            m.append(result)
        split_thres = last_result[m.index(min(m)) + 1][0] - 1
        split_thres_old = m.index(min(m)) + 1
        return split_thres, last_result[:split_thres_old], last_result[split_thres_old:]


class CART:
    def __init__(self, train_data: list, max_depth):
        self.train_data = train_data  # [(x1, y1), (x2, y2) ... ]
        self.max_depth = max_depth
        self.gbdtree = None

    def train(self):
        self.gbdtree = GBDTreeNode(0, self.max_depth, self.train_data)

    def predict(self, x):
        def find_children_node(root):
            # print(root.average, root.split_res)
            if not root.left and not root.right:
                return root.average
            if x <= root.split_res:
                if root.left:
                    return find_children_node(root.left)
                else:
                    return root.average
            else:
                if root.right:
                    return find_children_node(root.right)
                else:
                    return root.average
        return find_children_node(self.gbdtree)

    def print_tree(self):
        def print_tree_node(node):
            print(node.average, node.split_res)
            if node.left:
                print_tree_node(node.left)
            if node.right:
                print_tree_node(node.right)
        print_tree_node(self.gbdtree)


if __name__ == '__main__':
    train_set = [[1, 4.5], [2, 4.75], [3, 4.91], [4, 5.34], [5, 5.80], [6, 7.05], [7, 7.90], [8, 8.23], [9, 8.70], [10, 9.00]]
    gbdt = CART(train_set, 1)
    gbdt.train()
    gbdt.print_tree()
    print(gbdt.predict(3))
    for i in train_set:
        print(gbdt.predict(i[0]) == i[1])
    # gbdt.print_tree()