import numpy as np
from scipy.spatial.distance import cdist


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)
        
        print('self.train_y.dtype', self.train_y.dtype)
        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        #print('self.train_X.shape', self.train_X.shape)
        #print('X.shape', X.shape)
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        #print(dists.shape)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test, i_train] = np.abs(X[i_test] - self.train_X[i_train]).sum()
                pass
        
        return dists
     

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        #print('self.train_X.shape', self.train_X.shape)
        #print('X', X.shape)
       # print('self.train_X', self.train_X[0][0])
        #print('X', X[0][0])
        num_test = X.shape[0]
        #print('num_test', num_test)
        #dists = np.abs(self.train_X[0][0] - X[0][0])
        #print('dists', dists)
        dists = np.zeros((num_test, num_train), np.float32)
        #print('dists.shape', dists.shape)
        #print('X[0]', X[0])
        #print('self.train_X', self.train_X)
        #print('self.train_X - X[0]', self.train_X - X[0])
        
        for i_test in range(num_test):
            dists[i_test] = np.abs(self.train_X - X[i_test]).sum(axis = 1)
            #print('dists2', dists)
            # without additional loops or list comprehensions
            pass
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        #print('self.train_X', self.train_X)
        #print('self.train_X', self.train_X.shape)
        #print('self.train_X.T', self.train_X)
        num_test = X.shape[0]
        #print('X', X.shape)
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        dists = cdist(X, self.train_X, metric='cityblock')
        #print('dists', dists)
        #dists = np.abs(self.train_X.T[0] - X[0])
        #print(dists)
        # TODO: Implement computing all distances with no loops!
        pass
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        #print('dists', dists)
        #print('dists[0]', dists[0])
        #print('np.min(dists[0])', np.min(dists[0]))
        #print('np.argmin(dists[0])', np.argmin(dists[0]))
        #print('self.train_y np.argmin(dists[0])',self.train_y[np.argmin(dists[0])])
        #print('num_test', num_test)
        #print('self.train_y', self.train_y)
        #print('self.train_y.shape', self.train_y.shape)
        pred = np.zeros(num_test, np.bool)
        #print('pred', pred)
        for i in range(num_test):
            idx = np.argpartition(dists[i], self.k) # находим срез, в котором все элементы упорядочены по возрастанию и k наименьших стоят в начале
            counts = np.bincount(self.train_y[idx[:self.k]]) # берем срез первых k наименьших элементов. bincount возвращает массив counts, в котором индекс каждого элемента равен значению из self.train_y, а значение каждого элемента массива counts это сколько раз элемент встречается в self.train_y
            pred[i] = np.argmax(counts).astype(bool) # берем индекс элемента который встречается максимальное число раз, а индекс это 0 или 1, потому что у нас в self.train_y только True и False и мы проверяли, сколько раз встречается True и False. Индекс это и есть значение из self.train_y (True и False). Переводим индекс к типу bool
            #print('ssssssss', pred)
            #print('counts)', counts)
            #print('np.argmax(counts)', np.argmax(counts))
            pass
        
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            idx = np.argpartition(dists[i], self.k)
            counts = np.bincount(self.train_y[idx[:self.k]])
            pred[i] = np.argmax(counts)
            # TODO: Implement choosing best class based on k
            # nearest training samples
            pass
        return pred
