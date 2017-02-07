import Network
import mnist_loader


net = Network.NeuralNetwork([784,30,10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net.stochasticGradientDescent(training_data,30,10,1.0,test_data)
