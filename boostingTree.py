from CART_cut import CART


class BoostingTree:
    def __init__(self, max_depth, train_data):
        self.max_depth = max_depth
        self.source_train_data = train_data
        self.sub_trees = list()

    def train(self):
        train_data = self.source_train_data
        for i in range(100):
            print(train_data)
            cart = CART(train_data, self.max_depth)
            cart.train()
            predict = [cart.predict(i[0]) for i in train_data]
            train_data = [[train_data[i][0], train_data[i][1] - predict[i]] for i in range(len(train_data))]
            self.sub_trees.append(cart)

    def predict(self, x):
        result = 0
        for cart in self.sub_trees:
            result = result + cart.predict(x)
        return result

    def print_subTree(self):
        for cart in self.sub_trees:
            print(cart.gbdtree.split_res)


if __name__ == '__main__':
    train_set = [[1, 4.5], [2, 4.75], [3, 4.91], [4, 5.34], [5, 5.80], [6, 7.05], [7, 7.90], [8, 8.23], [9, 8.70],
                 [10, 9.00]]
    bt = BoostingTree(1, train_set)
    bt.train()
    for i in train_set:
        print(bt.predict(i[0]))
    print(bt.predict(2.1))
    print(bt.print_subTree())