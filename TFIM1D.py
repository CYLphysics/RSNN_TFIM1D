import numpy as np
import scipy as sp
from itertools import product
from numpy import linalg as LA
import math
import tensorflow as tf 
from IPython.display import clear_output
# if version of tensorflow >= 2.0:
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 

import matplotlib.pyplot as plt 
import time
import random
import os

class ISING_1D():
    def __init__(self, N_spin, hfield = 0, J = 1):
    
        self.N_spin    = N_spin
        self.hfield    = hfield
        self.J         = J

    def sampled_basis(self, bases_per_sample):
        self.bases_per_sample = bases_per_sample
        basis_sample = []
        for i in range(self.bases_per_sample):
            basis_sample.append(
                list(np.random.choice([1,-1], size = self.N_spin,replace = True))
            )
        return basis_sample

    def Ising_1D_Sample(self, bases_per_sample, basis_sample, noise = 0): 
        # This version can make computaional time increase linearly !!!!
        self.noise     = noise
        self.bases_per_sample = bases_per_sample
        bsnum = 0 
        H = np.zeros((self.bases_per_sample, self.bases_per_sample))
        for H_i in range(self.bases_per_sample):
            for H_j in range(self.bases_per_sample):
                H_sum = 0
                for i in range(self.N_spin):
                    if H_i == H_j:
                        if i == self.N_spin-1:          #OBC
                            H_sum -= 0
                        else:
                            H_sum -= self.J*basis_sample[H_j][i]*basis_sample[H_j][i+1] *(1+np.random.rand()*self.noise)

                    sj = list(basis_sample[H_j])
                    sj[i] *= -1
                    if tuple(sj) in basis_sample:
                        if H_i == basis_sample.index(tuple(sj)):
                            H_sum -= self.hfield*(1+np.random.rand()*self.noise)
                H[H_i, H_j] = H_sum
        return H, basis_sample

    def Ising_1D(self, PBC = True): #don't try too large
        self.PBC = PBC 
        basis = list(product([-1,1],repeat=self.N_spin))
        H = np.zeros((2**self.N_spin,2**self.N_spin))
        for H_i in range(2**self.N_spin):
            for H_j in range(2**self.N_spin):
                H_sum = 0
                for i in range(self.N_spin):
                    if H_i == H_j:
                        if PBC == True:      
                            if i == self.N_spin-1:
                                H_sum -= self.J*basis[H_j][i]*basis[H_j][0]
                            else:
                                H_sum -= self.J*basis[H_j][i]*basis[H_j][i+1]
                        elif PBC == False:
                            if i == self.N_spin-1:
                                H_sum -= 0
                            else:
                                H_sum -= self.J*basis[H_j][i]*basis[H_j][i+1]   
                    sj = list(basis[H_j])
                    sj[i] *= -1
                    if H_i == basis.index(tuple(sj)):
                        H_sum -= self.hfield
                H[H_i,H_j] = H_sum
        return H
    def JW_spectrum(self):
        self.L = self.N_spin
        self.h = self.hfield
        B = np.zeros([self.L, self.L])
        for i in range(self.L-1):
            B[i][i] = -2*self.h
            #B[i+1][i] = -self.t-self.delta
            B[i][i+1] = -2*self.J
        B[self.L-1][self.L-1] = -2*self.h
        E = np.linalg.svd(B)[1]
        E_list = [] 
        E_list.append(-1*np.sum(E)/2)
        for i in range(2):
            E_list.append(-1*np.sum(E)/2 + np.sort(E)[i])
        return E_list

    class Data_gen():
        
        def __init__(self, N_spin, bases_per_sample, N_sample_matricies = 5,
                 N_training_data_per_side = 5):

            self.N_spin = N_spin
            self.bases_per_sample = bases_per_sample
            self.N_sample_matricies = N_sample_matricies
            self.N_training_data_per_side = N_training_data_per_side
            
           
            B_n = self.bases_per_sample*self.N_sample_matricies
            basis_set_data = ISING_1D(self.N_spin,0,0).sampled_basis(B_n) 
            self.basis_set_data = basis_set_data

            print("Shape of basis_set_data =\n",np.shape(basis_set_data))

        def training_data(self, left_h = 0.3, right_h = 0.3):
            start = time.time()

            self.left_h = left_h
            self.right_h = right_h

            IS_sample_training_set = []
            IS_sample_training_set_ans = []
            print("\n###____Training data generation start!____###\n")
            
            if self.left_h != 0:
                for i in range(self.N_training_data_per_side+1):
                    s_ = []
                    for j in range(self.N_sample_matricies):
                        s_.append(
                                ISING_1D(self.N_spin,
                            0+(self.left_h/self.N_training_data_per_side)*i,
                            1-(self.left_h/self.N_training_data_per_side)*i
                            ).Ising_1D_Sample(
                                    self.bases_per_sample,
                                self.basis_set_data[
                                        0+self.bases_per_sample*j:self.bases_per_sample+self.bases_per_sample*j
                                        ])[0])
                    IS_sample_training_set.append(s_)
                    IS_sample_training_set_ans.append(
                        ISING_1D(self.N_spin,
                            0+(self.left_h/self.N_training_data_per_side)*i,
                            1-(self.left_h/self.N_training_data_per_side)*i
                            ).JW_spectrum()
                    )
                    
                    #end = time.time()
                    if i % 10 == 0:
                        print("training_set LHS, Processing ",i,"/",self.N_training_data_per_side, \
                        ", duration = ", round(time.time() - start, 5), "s")
            elif self.left_h == 0:
                print("No data will be generated in LHS !!")

            if self.right_h != 0:
                for i in range(self.N_training_data_per_side+1):
                    s_ = []
                    for j in range(self.N_sample_matricies):
                        s_.append(
                        ISING_1D(self.N_spin,
                        1-(self.right_h/self.N_training_data_per_side)*i,
                        0+(self.right_h/self.N_training_data_per_side)*i
                        ).Ising_1D_Sample(
                                self.bases_per_sample,
                                self.basis_set_data[
                                    0+self.bases_per_sample*j:self.bases_per_sample+self.bases_per_sample*j
                                    ])[0])
                    IS_sample_training_set.append(s_)
                    if i % 10 == 0:
                        print("training_set RHS, Processing ",i,"/",self.N_training_data_per_side, \
                        ", duration = ", round(time.time() - start, 5), "s")
                    IS_sample_training_set_ans.append(
                                ISING_1D(self.N_spin,
                        1-(self.right_h/self.N_training_data_per_side)*i,
                        0+(self.right_h/self.N_training_data_per_side)*i
                        ).JW_spectrum()
                    )
            elif self.right_h == 0:
                print("No data will be generated in RHS !!")
                
            print("\n###____Training data generation end!____###\n")
            print("\n###____Result____###\n")

            print("Shape of training data set = \n",np.shape(IS_sample_training_set))
            print("Shape of training label set = \n",np.shape(IS_sample_training_set_ans))

            if os.path.isdir("./TrainData") == False:
                os.makedirs("TrainData")

            ts_file_name = "./TrainData/IS"+str(self.N_spin)+ \
            "_sample_training_set_0"+str(int(100*self.left_h))+ \
            "0"+str(int(100*self.right_h))+ \
            "M_"+str(self.bases_per_sample)+".dat"

            tls_file_name = "./TrainData/IS"+str(self.N_spin)+ \
            "_sample_training_set_ans_0"+str(int(100*self.left_h))+ \
            "0"+str(int(100*self.right_h))+ \
            "M_"+str(self.bases_per_sample)+".dat"

            np.array(IS_sample_training_set).dump(ts_file_name)

            print("Save training data set as \n", ts_file_name
            )

            np.array(IS_sample_training_set_ans).dump(tls_file_name)

            print("Save training label set as \n",tls_file_name
            )
            
            return ts_file_name, tls_file_name

        def testing_data(self, N_testing_data = 100):
            start = time.time()
            self.N_testing_data = N_testing_data

            IS_sample_testing_set = []
            IS_sample_testing_set_ans = []

            print("\n###____Testing data generation start!____###\n")

            for i in range(self.N_testing_data+1):
                s_ = []
                for j in range(self.N_sample_matricies):
                    s_.append(
                            ISING_1D(self.N_spin,
                        0+(1/self.N_testing_data)*i,
                        1-(1/self.N_testing_data)*i
                        ).Ising_1D_Sample(
                                self.bases_per_sample,
                            self.basis_set_data[
                                    0+self.bases_per_sample*j:self.bases_per_sample+ \
                                    self.bases_per_sample*j
                                    ])[0])
                IS_sample_testing_set.append(s_)
                IS_sample_testing_set_ans.append(
                    ISING_1D(self.N_spin,
                        0+(1/self.N_testing_data)*i,
                        1-(1/self.N_testing_data)*i
                        ).JW_spectrum()
                )
                if i%10 == 0:
                    print("testing_set, Processing ",i,"/",self.N_testing_data,\
                    ", duration = ", round(time.time() - start, 5), "s")

            print("\n###____Testing data generation end!____###\n")
            print("\n###____Result____###\n")
            print("Shape of testing data set = \n",np.shape(IS_sample_testing_set))
            print("Shape of testing label set = \n",np.shape(IS_sample_testing_set_ans))
            if os.path.isdir("./TrainData") == False:
                os.makedirs("TrainData")

            ts_file_name = "./TrainData/IS"+str(self.N_spin)+"_sample_testing_set_0"\
            +str(int(100*self.left_h))+"0"+str(int(100*self.right_h))+"M_"\
            +str(self.bases_per_sample)+".dat"

            tsl_file_name = "./TrainData/IS"+str(self.N_spin)+"_sample_testing_set_ans_0"\
            +str(int(100*self.left_h))+"0"+str(int(100*self.right_h))+"M_"\
            +str(self.bases_per_sample)+".dat"        


            np.array(IS_sample_testing_set).dump(ts_file_name)

            print("Save testing data set as \n",
            ts_file_name
            )

            np.array(IS_sample_testing_set_ans).dump(tsl_file_name)

            print("Save testing label set as \n",
            tsl_file_name
            )

            return ts_file_name, tsl_file_name

                
class DATA_PRO:
    def Load_data(trainD_, ExactD_, testD_, ExactDtest_):
        testD      = np.load(testD_     ,allow_pickle=True)
        trainD     = np.load(trainD_    ,allow_pickle=True)
        ExactD     = np.load(ExactD_    ,allow_pickle=True)
        ExactDtest = np.load(ExactDtest_,allow_pickle=True)
        print("Shape of testD loaded =",np.shape(testD))
        print("Shape of trainD loaded =", np.shape(trainD))
        print("Shape of ExactD loaded = ", np.shape(ExactD))
        print("Shape of ExactDtest loaded = ",np.shape(ExactDtest))
        return  trainD, ExactD,testD, ExactDtest

    def Pick_eigs(num_of_eigs, ExactDtest_):
        ExactDtest_n = []
        c = 0
        for i in ExactDtest_:
            c += 1
            ExactDtest_n.append(i[0:num_of_eigs])
            
        print("The result shape of picked Eig_n = ",np.shape(ExactDtest_n))
        return ExactDtest_n
    def Indicies_Randomization(trainD_, ExactD_3_):
        trainD_ExactD_3 = list(zip(trainD_, ExactD_3_))
        random.shuffle(trainD_ExactD_3)
        trainD, ExactD_3 = zip(*trainD_ExactD_3)
        return trainD, ExactD_3


def Train_CNN(
    trainD,
    ExactD,
    testD_s,
    ExactDtest_s_3,
    batch_size = 5,
    batch_num  = 220,
    epoch_num  = 10,
    Channel1   = 10,
    Channel2   = 10,
    betaa      = 0.01,
    N_of_sample = 100,
    sample_size = 10
    ):
    start = time.time()
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    def conv2d(x, W):
    #logits = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #return conv_batch_norm(logits, n_out)
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    def shape_cal(x):
        return math.ceil((x - 4)/4 + 1)

    x       = tf.placeholder("float",[None , N_of_sample, sample_size, sample_size]) 
    # this placeholder is in the form [ N_of_sample , sample_size * sample_size]
    y       = tf.placeholder("float",[None , 3]) # The output size 
    x_image = tf.placeholder_with_default(
        tf.reshape(
            x, [
                -1, int(N_of_sample), sample_size*sample_size, 1
                ]), (
                    None, int(N_of_sample), sample_size*sample_size, 1
                    ) )


    W_conv1 = weight_variable([5, 5, 1, Channel1])
    b_conv1 = bias_variable([Channel1])
    h_conv1 = conv2d(x_image, W_conv1) + b_conv1
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, Channel1, Channel2])
    b_conv2 = bias_variable([Channel2])
    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_pool2 = max_pool_2x2(h_conv2)


    h_pool2_flat = tf.placeholder_with_default(
        tf.reshape(h_pool2,
         [
             -1 ,Channel2, int(shape_cal(N_of_sample)*shape_cal(sample_size*sample_size))
             ]), (
                 None, Channel2, int(shape_cal(N_of_sample)*shape_cal(sample_size*sample_size)))
                 )
        #N_of_sample/4 * sample_size*sample_size /4 * channel2

    W_fc1 = weight_variable([int(shape_cal(N_of_sample)*shape_cal(sample_size*sample_size)), Channel2])
    b_fc1 = bias_variable([Channel2])
    h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    W_fc2 = weight_variable([Channel2, int(shape_cal(N_of_sample)*shape_cal(sample_size*sample_size))])
    b_fc2 = bias_variable([int(shape_cal(N_of_sample)*shape_cal(sample_size*sample_size))])
    
    L1 = tf.matmul(h_fc1, W_fc2) + b_fc2
    L1p = tf.reshape(L1, [-1, Channel2*int(shape_cal(N_of_sample)*shape_cal(sample_size*sample_size))])
    w1 = weight_variable([Channel2*int(shape_cal(N_of_sample)*shape_cal(sample_size*sample_size)),10])
    b1 = bias_variable([10])
    L2 = tf.nn.selu(tf.matmul(L1p, w1) + b1)
    w2 = weight_variable([10,3])
    b2 = bias_variable([3])
    L3 = tf.matmul(L2, w2) + b2 # [-1, 1, 3]
    pred = tf.reshape(L3, [-1, 3])
    w = [W_conv1, W_conv2, W_fc1, W_fc2, w1, w2] #, w3, w4, w5, w6, w7, w8]
    b = [b_conv1, b_conv2, b_fc1, b_fc2, b1, b2] #, b3, b4, b5, b6, b7, b8]
    regularizerw = tf.reduce_mean([tf.nn.l2_loss(i) for i in w])
    regularizerb = tf.reduce_mean([tf.nn.l2_loss(i) for i in b])
    cost = tf.reduce_sum(tf.pow(pred-y,2) + betaa * tf.add(regularizerw,regularizerb))

    #cost = tf.pow(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels =  y)),2)

    acc  = (1 - tf.reduce_mean(tf.abs((y - pred)/y)))*100
    optimizer = tf.train.AdamOptimizer(epsilon=1e-1).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(epsilon = 1e-3).minimize(cost)
    
    #tf.train.GradientDescentOptimizer(1e-2).minimize(cost) #tf.train.AdamOptimizer(epsilon = 0.1).minimize(cost)
    testbatch = batch_size//3
    # this means we only apply 1 weight parameter update for an input
    coststep     = []
    testcoststep = []
    acc_array    = []
    acc_array_train = []
    acc_array_pred  = []
    bav_list = []
    c_list = []
    input_data  = trainD #[batch_idx:batch_idx+batch_size]
    output_data = ExactD #[batch_idx:batch_idx+batch_size]
    test_data   = testD_s #[batch_idx:batch_idx+testbatch]
    test_answ   = ExactDtest_s_3 #[batch_idx:batch_idx+testbatch]
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        cc = 0
        for epoch in range(epoch_num):
            batch_acc_sum = 0 
            for batch_idx in range(batch_num):
                time_per_step = time.time()
                opt, acc_train_, cost_ = sess.run([optimizer, acc, cost ],
                                                feed_dict={
                                                    x:input_data[batch_idx:batch_idx+batch_size],
                                                    y:output_data[batch_idx:batch_idx+batch_size]
                                                    }
                                                )
                acc_, cost_va = sess.run( [acc, cost],
                                feed_dict={
                                    x:test_data[[30,31,32,68,69,70]], #x:test_data[2490:2510],
                                    y:test_answ[[30,31,32,68,69,70]]  #y:test_answ[2490:2510]
                                    }
                                )
                acc_p = sess.run( acc, feed_dict={
                    x:test_data[32:68],
                    y:test_answ[32:68]

                })
                if batch_idx % 100 == 0:
                    clear_output()
                    print("epoch =",epoch+1,"/",epoch_num,"(",((epoch+1)/epoch_num)*100,"%)",
                        ",Batch =",batch_idx+1,"/",batch_num,"(",((batch_idx+1)/batch_num)*100,"%)",
                    "\n train accuracy = \n",
                    acc_train_,
                    "%"
                    )
                    print(
                    "\n Prediction accuracy = \n",
                    sess.run( acc,
                            feed_dict={
                                x:test_data[32:68],#x:test_data[2490:2510],
                                y:test_answ[32:68] #y:test_answ[2490:2510]
                                }
                            ),
                    "%"
                    )
                    print(
                    "\n validation accuracy = \n",
                    sess.run( acc,
                            feed_dict={
                                x:test_data[[30,31,32,68,69,70]],#x:test_data[2490:2510],
                                y:test_answ[[30,31,32,68,69,70]] #y:test_answ[2490:2510]
                                }
                            ),
                    "%"
                    )
                    print(
                        "\n Cost = \n",
                        cost_
                        )
                    
                    print(
                        "\n Validation Cost = \n",
                        cost_va
                        )    
                    dur = time.time()
                    print("Duration :",dur - start,"s")
                    print("tps :",dur - time_per_step,"s")
                    print("---------------------------------------------")

                    
                    
                coststep.append(cost_)
                acc_array.append(acc_)
                acc_array_train.append(acc_train_)
                acc_array_pred.append(acc_p)
                testcoststep.append(cost_va)

                #Early break

                batch_acc_sum += acc_array[epoch*batch_num+batch_idx]

            batch_acc_avg = batch_acc_sum/batch_num
            bav_list.append(batch_acc_avg)
            
            c = 0
            if epoch > 10: #int(epoch_num*0.3):
                for i in range(10):
                    if bav_list[epoch - i] > 99.7:
                    #if bav_list[epoch-i] < bav_list[epoch-i-1]:
                        c += 1
            c_list.append(c)
                    
            print("batch_acc_avg =",batch_acc_avg,"%")
            print("decrease count = ",c)
            print("---------------------------------------------")
            
            if c >= 10: # or 90
                print("break!")
                break;
    

        W_conv1, W_conv2, W_fc1, W_fc2, w1, w2, b_conv1, b_conv2, b_fc1, b_fc2, b1, b2 = sess.run(
                [W_conv1, W_conv2, W_fc1, W_fc2, w1, w2, b_conv1, b_conv2, b_fc1, b_fc2, b1, b2])
        Pred = sess.run( pred,feed_dict={x:test_data[0:101]})
        loctime = time.asctime( time.localtime(time.time()))
        print("Saveing data to ",str(loctime))
        dire = "./ResultData/"+str(loctime)
        os.makedirs(dire)
        os.chdir(dire)    
        #==========SAVE============
        np.array(Pred).dump("Pred.dat")
        np.array(coststep).dump("coststep.dat")
        np.array(testcoststep).dump("testcoststep.dat")
        np.array(acc_array).dump("acc_array.dat")
        np.array(acc_array_train).dump("acc_array_train.dat")
        np.array(acc_array_pred).dump("acc_array_pred.dat")
        np.array(c_list).dump("c_list.dat")
        np.array(bav_list).dump("bav_list.dat")
        W_conv1.dump("W_conv1.dat")
        W_conv2.dump("W_conv2.dat")
        W_fc1.dump("W_fc1.dat")
        W_fc2.dump("W_fc2.dat")
        w1.dump("w1.dat")
        w2.dump("w2.dat")
        b_conv1.dump("b_conv1.dat")
        b_conv2.dump("b_conv2.dat")
        b_fc1.dump("b_fc1.dat")
        b_fc2.dump("b_fc2.dat")
        b1.dump("b1.dat")
        b2.dump("b2.dat")

        end = time.time()
        print("Total execute time :",end - start,"s")
        Tet =  end - start
        np.array(Tet).dump("Tet.dat")
        #==========================
        #======back to previous dir=======
        os.chdir("..")
        os.chdir("..")

        return dire


def plot_res(folder, rh, lh, ExactDtest):
    from scipy.interpolate import make_interp_spline, BSpline
    from scipy.ndimage.filters import gaussian_filter1d

    Pred = np.load(folder+"/Pred.dat",allow_pickle=True)
    Cost = np.load(folder+"/coststep.dat",allow_pickle=True)
    Cost_test = np.load(folder+"/testcoststep.dat",allow_pickle=True)
    acc  = np.load(folder+"/acc_array.dat",allow_pickle=True)
    acc_t = np.load(folder+"/acc_array_train.dat",allow_pickle=True)
    acc_p = np.load(folder+"/acc_array_pred.dat",allow_pickle=True)
    c_list = np.load(folder+"/c_list.dat",allow_pickle=True)

    axes_ls = []
    for i in range(101):
        axes_ls.append(i/100)

    plt.style.use("science")

    fig = plt.figure(figsize=(5,3), dpi=120)
    ax = plt.axes()

    ax.axvspan(1-rh, 1, alpha=0.25, color=plt.cm.Oranges(0.3))
    ax.axvspan(0,lh, alpha=0.25, color=plt.cm.Oranges(0.3))
    ax.plot(axes_ls, (Pred.T)[0],color = plt.cm.RdBu(0.1),label="RSNN")
    ax.plot(axes_ls,(ExactDtest.T)[0],color = plt.cm.RdBu(0.9),label="Exact")
    ax.plot(axes_ls, Pred,color = plt.cm.RdBu(0.1))
    ax.plot(axes_ls,ExactDtest,color = plt.cm.RdBu(0.9))

    #plt.text(0.1,-7,"N = 20,\nh+J=1")
    #plt.ylim((-30,-17))
    plt.ylabel("$E/(J+h)$")
    plt.xlabel("$\lambda$")
    #plt.legend(loc = (0.327, 0.27),fontsize = 8)
    plt.legend()

    plt.show()

    Cost_test_epoch = []
    Cost_epoch = []
    acc_t_epoch = []
    acc_epoch = []
    acc_p_epoch = []

    for i in range(np.shape(Cost_test)[0]):
        if i % 204 == 0:
            Cost_test_epoch.append(Cost_test[i])
            Cost_epoch.append(Cost[i])
            acc_t_epoch.append(acc_t[i])
            acc_epoch.append(acc[i])
            acc_p_epoch.append(acc_p[i])

    plt.style.use("science")
    fig, ax2 = plt.subplots(figsize=(5,3), dpi=120)

    plt.yscale('log')
    ax1 = ax2.twinx()


    ax1.plot(gaussian_filter1d(acc_t, sigma=1.3),label = 'Train. Acc.',linewidth = 2 ,color=plt.cm.RdBu(0.1))
    ax1.plot(gaussian_filter1d(acc, sigma=1.3), label = 'Valid. Acc.',linewidth = 2,color=plt.cm.RdBu(0.3))
    ax1.plot(gaussian_filter1d(acc_p, sigma=1.3),label = 'Pred. Acc.',linewidth = 2,color=plt.cm.RdBu(0.4))
    ax2.plot(0,0)
    ax2.plot(0,0)
    ax2.plot(0,0)
    ax2.plot(gaussian_filter1d(Cost_test, sigma=1.3),label = 'Valid. Loss',linewidth = 2,color=plt.cm.RdBu(0.7))
    ax2.plot(gaussian_filter1d(Cost, sigma=1.3), label = 'Train. Loss',linewidth = 2,color=plt.cm.RdBu(0.9))

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax1.set_ylabel('Accuracy (\%)')

    ax2.set_ylim(bottom = 0, top = 1000)
    ax1.set_ylim(bottom = 0, top = 101)

    ax2.legend(frameon = True, loc = (0.6, 0.7))
    ax1.legend(frameon = True, loc = (0.6, 0.4))

    plt.show()