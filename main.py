from utils import LayerDense

X = [[1,2,3,2.5],
     [2,5,-1,2],
     [-1.5,2.7,3.3,-0.8]]

if __name__ == '__main__':


    layer1 = LayerDense(4, 5)
    layer2 = LayerDense(5, 2)

    layer1.forward(X)
    print(layer1.output)
    layer2.forward(layer1.output)
    print(layer2.output)




